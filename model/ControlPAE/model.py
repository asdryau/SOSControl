import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from pathlib import Path

from utils.utils import *
import math
from diffusers import DDIMScheduler, DDPMScheduler

# import diffusion
from model.ControlDiffusion.diffusion import ControlDiffusion

# import pae
from model.PAE.model import MotionPAE

loss_mse = nn.MSELoss()
loss_l1 = nn.L1Loss()

### reference: MLD
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)    #(1,T,E)

        self.register_buffer('pe', pe)

    def forward(self, x, placeholder=None):
        # X: (B,T,E)
        x = x + self.pe[:,:x.shape[1], :x.shape[2]]
        return self.dropout(x)
    
def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class HintBlock(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.ModuleList([
            nn.Linear(self.input_feats, self.latent_dim),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.Linear(self.latent_dim, self.latent_dim),
            zero_module(nn.Linear(self.latent_dim, self.latent_dim))
        ])

    def forward(self, x):
        x = x.permute((1, 0, 2))

        for module in self.poseEmbedding:
            x = module(x)  # [seqlen, bs, d]

        x = x.permute((1, 0, 2))
        return x
    
# replace forward
def replace_function(model, src,
            mask = None,
            src_key_padding_mask = None,
            control = None, return_intermediate=False):
    output = src
    intermediate = []

    for i, layer in enumerate(model.layers):
        output = layer(output, src_mask=mask,
                        src_key_padding_mask=src_key_padding_mask)
        if control is not None:
            output = output + control[i]

        if return_intermediate:
            intermediate.append(output)

    if return_intermediate:
        return torch.stack(intermediate)
    return output
    
class ControlMotionPAE(pl.LightningModule):
    def __init__(self, controlmdm_path, mdm_path, pae_path):
        super().__init__()

        self.mdm_path = mdm_path
        self.pae_path = pae_path
        self.save_hyperparameters()

        # ### !!! standardize rep
        self.rep_mean = torch.from_numpy(np.load(f"./model/rep10_mean.npy"))
        self.rep_std = torch.from_numpy(np.load(f"./model/rep10_std.npy"))

        self.mdm_model = ControlDiffusion.load_from_checkpoint(controlmdm_path, mdm_path=mdm_path)
        pae_model = MotionPAE.load_from_checkpoint(pae_path)

        self.mdm_model = self.mdm_model.eval()
        for param in self.mdm_model.parameters():
            param.requires_grad = False

        # -- PAE --
        # model setting
        self.input_dim = pae_model.input_dim
        self.latent_dim = pae_model.latent_dim

        self.gmd_proj, self.gmd_proj_inv = pae_model.gmd_proj, pae_model.gmd_proj_inv
        self.sequence_pos_encoder = pae_model.sequence_pos_encoder

        self.enc_query = pae_model.enc_query
        self.skelEmbedding = pae_model.skelEmbedding
        self.encoder = pae_model.encoder

        self.decoder = pae_model.decoder
        self.finallayer = pae_model.finallayer

        # -- CPAE --
        self.placeholder_token = nn.Embedding(1, 18)
        self.input_hint_block = HintBlock(18, self.latent_dim)
        self.c_sequence_pos_encoder = PositionalEncoding(self.latent_dim, 0.1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.latent_dim, nhead=8, dim_feedforward=1024, dropout=0.1, batch_first=True)
        self.c_decoder = nn.TransformerEncoder(encoder_layer, num_layers=8) # return_intermediate???
        self.zero_convs = zero_module(nn.ModuleList([nn.Linear(self.latent_dim, self.latent_dim) for _ in range(8)]))

    def transform(self, data):
        data = (data - self.rep_mean.to(data.device)) / self.rep_std.to(data.device)
        return data
    
    def inv_transform(self, data):
        data = data * self.rep_std.to(data.device) + self.rep_mean.to(data.device)
        return data
    
    def latent_reparam(self, latents, progress):
        f = latents[:,0:1]     #(B,1,?)
        a = latents[:,1:2]     #(B,1,?)
        b = latents[:,2:3]     #(B,1,?)
        s = latents[:,3:4]     #(B,1,?)

        ### scale progress
        left_valid = (progress < 0).float()      #(B,T) 
        right_valid = (progress > 0).float()
        T_left = torch.sum(left_valid, dim=1, keepdim=True) 
        T_right = torch.sum(right_valid, dim=1, keepdim=True) 
        # length = T_right + T_left + 1

        progress_2 = (progress * left_valid * T_left + progress * right_valid * T_right).detach()

        ### calc F, T
        progress = progress.unsqueeze(-1)  # (B,T) --> (B,T,1)
        progress_2 = progress_2.unsqueeze(-1)

        reparam_latents_1 = a * torch.sin(f * progress + s) + b         #(B,3,?) --> (B,T,?)
        reparam_latents_2 = a * torch.sin(f * progress_2 + s) + b         #(B,3,?) --> (B,T,?)

        reparam_latents = torch.cat([reparam_latents_1[:,:,0::2], reparam_latents_2[:,:,1::2]], dim=2)
        return reparam_latents   

    def encode(self, motion_rot, len_mask, progress):
        # (B,T=?,F=24*6), (B,T)
        B,T = motion_rot.shape[:2]
        # process motion
        motion_vec = self.skelEmbedding(motion_rot) 
        motion_vec = self.sequence_pos_encoder(motion_vec, progress)      #(B,T,E)

        ### generate learned token for ABS extraction
        queries = torch.arange(4, device=motion_rot.device).unsqueeze(0).long()    #(1,4)
        queries = queries.expand((B,4)).reshape((B*4))                              #(B*4,)
        queries = self.enc_query(queries).reshape((B,4,-1)) 

        # combine learned token to motion
        motion_seq = torch.cat((queries, motion_vec), dim=1)    #(B,T+4,E)
        token_mask = torch.ones((B, 4), dtype=bool, device=len_mask.device)
        len_mask = torch.cat((token_mask, len_mask), dim=1)
        # enc
        motion_seq = self.encoder(motion_seq, src_key_padding_mask=~len_mask)

        latents = motion_seq[:,:4]
        return latents         #(B,4,E)
    
    def c_decode(self, reparam_latents, hint, hint_masks, len_mask, progress, weight=1.0):
        # (B,T,?), (B,T)          

        # hint_masks = hint.sum(-1) != 0
        hint_mask_placeholder = self.placeholder_token(torch.zeros(1, device=hint.device, dtype=torch.long))
        hint = hint * hint_masks.float() + hint_mask_placeholder.unsqueeze(0) * (1-hint_masks.float())
        guided_hint = self.input_hint_block(hint.float())

        reparam_latents = reparam_latents + guided_hint
        reparam_latents = self.sequence_pos_encoder(reparam_latents, progress)

        # decoder 
        # output = self.decoder(reparam_latents, src_key_padding_mask=~len_mask)    #(B,T,E)
        output = replace_function(model=self.c_decoder, src=reparam_latents, return_intermediate=True, src_key_padding_mask=~len_mask)
        control = []
        for i, module in enumerate(self.zero_convs):
            control.append(module(output[i]))
        control = torch.stack(control)

        control = control * weight
        return control

    def decode(self, reparam_latents, control, len_mask, progress):
        # (B,T,?), (B,T)          
        # decoder 
        reparam_latents = self.sequence_pos_encoder(reparam_latents, progress)
        # output = self.decoder(reparam_latents, src_key_padding_mask=~len_mask)    #(B,T,E)
        output = replace_function(model=self.decoder, src=reparam_latents, src_key_padding_mask=~len_mask, control=control)
        output = self.finallayer(output)                                                        #(B,T,F=24*6)

        ### pad to fix the predicted length
        output = torch.nn.functional.pad(output, (0,0,0,len_mask.shape[1]-output.shape[1]))

        # output = torch.nn.functional.pad(output, (0,0,0,len_mask.shape[1]-output.shape[1]))
        output = output * len_mask.float().unsqueeze(-1)
        return output
        
    def configure_optimizers(self):
        opt_0 = torch.optim.AdamW(self.parameters(), lr=1e-4)
        return opt_0

    def loss_rec(self, motion_rec, motion_rot, len_mask):
        # (B,T=?,F=24*6)
        B,T,F = motion_rec.shape
        len_mask = len_mask.float()
        
        # mask
        motion_rec = motion_rec * len_mask.unsqueeze(-1)
        motion_rot = motion_rot * len_mask.unsqueeze(-1)
        # reshape
        motion_rec = motion_rec.reshape((B*T,-1))            #(B*T,J*6)
        motion_rot = motion_rot.reshape((B*T,-1))            #(B*T,J*6)
        len_mean = len_mask.reshape((-1)).mean()    #(B*T,)

        ### calc loss
        # recon loss on rot
        loss_rot = loss_mse(motion_rec, motion_rot) / len_mean
        return loss_rot


    def training_step(self, train_batch, batch_idx):
        mode = 'train'

        train_batch = [b.detach() for b in train_batch]
        id, motions, rep, hint, hint_weight, texts, masks, progress = train_batch

        B,T = rep.shape[:2]

        motions = torch.matmul(motions, self.gmd_proj.to(motions.device))

        rep = self.transform(rep)

        # timestep
        t = torch.randint(0, self.mdm_model.num_train_timesteps, (B, ), device=rep.device, dtype=torch.long).detach()

        # order mask hint   (order in (B,T,5))
        salient_thresh = torch.rand(B, hint_weight.shape[2], device=rep.device).detach()
        hint_mask = (hint_weight >= salient_thresh.unsqueeze(1))

        hint_mask = hint_mask.unsqueeze(-1).expand((-1,-1,-1,3))
        hint_mask = hint_mask.reshape((B,T,18))

        hint = hint.reshape((B,T,18))

        ############################
        # text_emb
        text_idx = torch.randint(0, 12, (B,), device=rep.device, dtype=torch.long).detach()
        texts = texts[torch.arange(B,device=rep.device, dtype=torch.long), text_idx]

        

        # bp_code (tgt)
        rep_noise = torch.randn_like(rep)
        rep_t = self.mdm_model.noise_scheduler.add_noise(rep.clone().cpu(), rep_noise.cpu(), t.cpu()).detach().to(rep.device)

        ############
        ###    diffusion step
        ############
        control = self.mdm_model.cmdm_condition_step(t, rep_t, hint, hint_mask, texts, masks, progress)
        rep_0 = self.mdm_model.mdm_diffusion_step(t, rep_t, control, texts, masks, progress)
        rep = rep_0.detach()
        #################################
        rep = self.inv_transform(rep)

        control = self.c_decode(rep,  hint, hint_mask, masks, progress)
        rec = self.decode(rep, control, masks, progress)

        loss = self.loss_rec(rec, motions, masks)

        self.log(f"{mode}/loss", loss)
        return loss


    def validation_step(self, train_batch, batch_idx):
        mode = 'valid'
        train_batch = [b.detach() for b in train_batch]
        id, motions, rep, hint, hint_weight, texts, masks, progress = train_batch

        B,T = rep.shape[:2]

        motions = torch.matmul(motions, self.gmd_proj.to(motions.device))

        rep = self.transform(rep)

        # timestep
        t = torch.randint(0, self.mdm_model.num_train_timesteps, (B, ), device=rep.device, dtype=torch.long).detach()
        
        # order mask hint   (order in (B,T,5))
        salient_thresh = torch.rand(B, hint_weight.shape[2], device=rep.device).detach()
        hint_mask = (hint_weight >= salient_thresh.unsqueeze(1))

        hint_mask = hint_mask.unsqueeze(-1).expand((-1,-1,-1,3))
        hint_mask = hint_mask.reshape((B,T,18))

        hint = hint.reshape((B,T,18))

        ############################
        # text_emb
        text_idx = torch.randint(0, 12, (B,), device=rep.device, dtype=torch.long).detach()
        texts = texts[torch.arange(B,device=rep.device, dtype=torch.long), text_idx]


        # bp_code (tgt)
        rep_noise = torch.randn_like(rep)
        rep_t = self.mdm_model.noise_scheduler.add_noise(rep.clone().cpu(), rep_noise.cpu(), t.cpu()).detach().to(rep.device)

        ############
        ###    diffusion step
        ############
        control = self.mdm_model.cmdm_condition_step(t, rep_t, hint, hint_mask, texts, masks, progress)
        rep_0 = self.mdm_model.mdm_diffusion_step(t, rep_t, control, texts, masks, progress)
        rep = rep_0.detach()
        #################################
        rep = self.inv_transform(rep)

        control = self.c_decode(rep, hint, hint_mask, masks, progress)
        rec = self.decode(rep, control, masks, progress)

        loss = self.loss_rec(rec, motions, masks)

        self.log(f"{mode}/loss", loss)
        return loss

    # implemented in eval script
    def test_step(self, train_batch, batch_idx):
        mode = 'test'
        pass
