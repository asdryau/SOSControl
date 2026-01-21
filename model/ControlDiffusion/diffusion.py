import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from pathlib import Path

from utils.utils import *
import math
from diffusers import DDIMScheduler, DDPMScheduler

# import diffusion
from model.Diffusion.diffusion import Diffusion

loss_mse = nn.MSELoss()
loss_l1 = nn.L1Loss()

### reference: MLD
def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param embedding_dim: the dimension of the output. :param max_period: controls the minimum frequency of the
    embeddings. :return: an [N x dim] Tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

class TimestepEmbedding(nn.Module):
    def __init__(self, channel: int, time_embed_dim: int, act_fn: str = "silu"):
        super().__init__()

        self.linear_1 = nn.Linear(channel, time_embed_dim)
        self.act = None
        if act_fn == "silu":
            self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, sample):
        sample = self.linear_1(sample)

        if self.act is not None:
            sample = self.act(sample)

        sample = self.linear_2(sample)
        return sample

class Timesteps(nn.Module):
    def __init__(self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

    def forward(self, timesteps):
        t_emb = get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
        )
        return t_emb

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
    
class ControlDiffusion(pl.LightningModule):
    def __init__(self, mdm_path):
        super().__init__()

        self.num_train_timesteps = 1000
        self.num_inference_timesteps = 1000
        self.mdm_path = mdm_path
        self.save_hyperparameters()
        
        mdm_model = Diffusion.load_from_checkpoint(mdm_path)

        # -- MDM --
        # model setting
        self.latent_dim = mdm_model.latent_dim

        self.gmd_proj, self.gmd_proj_inv = mdm_model.gmd_proj, mdm_model.gmd_proj_inv
        self.sequence_pos_encoder = mdm_model.sequence_pos_encoder

        self.enc_query = mdm_model.enc_query
        # self.linear_motin = mdm_model.linear_motin
        # self.linear_motout = mdm_model.linear_motout
        self.text_embedding = mdm_model.text_embedding

        self.scheduler = mdm_model.scheduler
        self.noise_scheduler = mdm_model.noise_scheduler
        self.time_proj = mdm_model.time_proj
        self.time_embedding = mdm_model.time_embedding
        self.denoiser = mdm_model.denoiser


        # -- CMDM --
        self.placeholder_token = nn.Embedding(1, 18)
        self.input_hint_block = HintBlock(18, self.latent_dim)
        # self.c_linear_motin = nn.Linear(269, self.latent_dim)
        self.c_sequence_pos_encoder = PositionalEncoding(self.latent_dim, 0.1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.latent_dim, nhead=8, dim_feedforward=1024, dropout=0.1, batch_first=True)
        self.c_seqTransEncoder = nn.TransformerEncoder(encoder_layer, num_layers=8) # return_intermediate???
        self.zero_convs = zero_module(nn.ModuleList([nn.Linear(self.latent_dim, self.latent_dim) for _ in range(8)]))
        self.c_embed_timestep = TimestepEmbedding(512, self.latent_dim)
        self.c_embed_text = nn.Linear(512, self.latent_dim)


    
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

    def denoise(self, time_emb, motions_t, text_emb, len_mask, control):
        B = motions_t.shape[0]
        # with torch.set_grad_enabled(True):
        #     motions_t.requires_grad_(True)
            ##########
            #   bp part
            
        #########
        # mot part
        queries = torch.arange(2, device=motions_t.device).unsqueeze(0).long()    #(1,11)
        queries = queries.expand((B,2)).detach()                             #(B,11)
        queries = self.enc_query(queries)

        time_token = queries[:,0:1] + time_emb
        text_token = queries[:,1:2] + text_emb

        # mask
        mot_mask = torch.ones((B, 2), dtype=bool, device=motions_t.device)
        mot_mask = torch.cat((mot_mask, len_mask), dim=1)

        ### concat into sequence
        mot_seq_t = torch.cat((time_token, text_token, motions_t), dim=1)    #(B,T+3,E)
        #self=self.denoiser,
        out_seq = replace_function(model=self.denoiser, src=mot_seq_t, src_key_padding_mask=~mot_mask, control=control)
        return out_seq[:,2:]               #(B,4,E)
       
    def mdm_diffusion_step(self, t, motions_t, control, texts, masks, progress):
            # time emb and text emb
        time_emb = self.time_proj(t).type_as(motions_t)
        time_emb = self.time_embedding(time_emb).unsqueeze(1)               #(B,1,E)

        text_emb = self.text_embedding(texts).unsqueeze(1)                  #(B,1,E)

        ### diffusion training
            # expand to latent sine wave
        # motion
        # motions_t = self.linear_motin(motions_t)
        motions_t = self.sequence_pos_encoder(motions_t, progress)

        motions_0 = self.denoise(time_emb, motions_t, text_emb, masks, control)
        # motions_0 = self.linear_motout(motions_0)
        return motions_0
    
    def cmdm_condition_step(self, t, motions_t, hint, hint_masks, texts, masks, progress, weight=1.0):
        time_emb = self.time_proj(t).type_as(motions_t)
        time_emb = self.c_embed_timestep(time_emb).unsqueeze(1)               #(B,1,E)
        text_emb = self.c_embed_text(texts).unsqueeze(1)                  #(B,1,E)
        
        B,T = masks.shape
        queries = torch.arange(2, device=motions_t.device).unsqueeze(0).long()    #(1,11)
        queries = queries.expand((B,2)).detach()                             #(B,11)
        queries = self.enc_query(queries)

        time_token = queries[:,0:1] + time_emb
        text_token = queries[:,1:2] + text_emb

        # hint_masks = hint.sum(-1) != 0
        hint_mask_placeholder = self.placeholder_token(torch.zeros(1, device=hint.device, dtype=torch.long))
        hint = hint * hint_masks.float() + hint_mask_placeholder.unsqueeze(0) * (1-hint_masks.float())
        guided_hint = self.input_hint_block(hint.float())

        # motions_t = self.c_linear_motin(motions_t)
        motions_t = motions_t + guided_hint
        # xseq = torch.cat((time_emb, text_emb, motions_t), axis=1)
        xseq = self.sequence_pos_encoder(motions_t, progress)
        xseq = torch.cat((time_token, text_token, xseq), axis=1)
        
        # mask
        B = motions_t.shape[0]
        mot_mask = torch.ones((B, 2), dtype=bool, device=motions_t.device)
        mot_mask = torch.cat((mot_mask, masks), dim=1)

        # print(xseq.shape, mot_mask.shape)   #self=self.c_seqTransEncoder,
        output = replace_function(model=self.c_seqTransEncoder, src=xseq, return_intermediate=True, src_key_padding_mask=~mot_mask)
        control = []
        for i, module in enumerate(self.zero_convs):
            control.append(module(output[i]))
        control = torch.stack(control)

        control = control * weight
        return control


    def training_step(self, train_batch, batch_idx):
        mode = 'train'

        train_batch = [b.detach() for b in train_batch]
        id, motions, hint, hint_weight, texts, masks, progress = train_batch
        B,T = masks.shape


        # timestep
        t = torch.randint(0, self.num_train_timesteps, (B, ), device=motions.device, dtype=torch.long).detach()

        # order mask hint   (order in (B,T,5))
        salient_thresh = torch.rand(B, hint_weight.shape[2], device=motions.device).detach()
        hint_mask = (hint_weight >= salient_thresh.unsqueeze(1))

        hint_mask = hint_mask.unsqueeze(-1).expand((-1,-1,-1,3))
        hint_mask = hint_mask.reshape((B,T,18))

        hint = hint.reshape((B,T,18))

        # motions = torch.matmul(motions, self.gmd_proj.to(motions.device))

        # text_emb
        text_idx = torch.randint(0, 12, (B,), device=motions.device, dtype=torch.long).detach()
        texts = texts[torch.arange(B,device=motions.device, dtype=torch.long), text_idx]


        # bp_code (tgt)
        motions_noise = torch.randn_like(motions)
        motions_t = self.noise_scheduler.add_noise(motions.clone().cpu(), motions_noise.cpu(), t.cpu()).detach().to(motions.device)

        ############
        ###    diffusion step
        ############
        control = self.cmdm_condition_step(t, motions_t, hint, hint_mask, texts, masks, progress)
        motions_0 = self.mdm_diffusion_step(t, motions_t, control, texts, masks, progress)

        ### total loss
        loss = self.loss_rec(motions_0, motions, masks)
        self.log(f"{mode}/loss", loss)
        return loss

    def validation_step(self, train_batch, batch_idx):
        mode = 'valid'

        train_batch = [b.detach() for b in train_batch]
        id, motions, hint, hint_weight, texts, masks, progress = train_batch
        B,T = masks.shape

        # timestep
        t = torch.randint(0, self.num_train_timesteps, (B, ), device=motions.device, dtype=torch.long).detach()

        # order mask hint   (order in (B,T,5))
        salient_thresh = torch.rand(B, hint_weight.shape[2], device=motions.device).detach()
        hint_mask = (hint_weight >= salient_thresh.unsqueeze(1))

        hint_mask = hint_mask.unsqueeze(-1).expand((-1,-1,-1,3))
        hint_mask = hint_mask.reshape((B,T,18))

        hint = hint.reshape((B,T,18))

        # motions = torch.matmul(motions, self.gmd_proj.to(motions.device))

        # text_emb
        text_idx = torch.randint(0, 12, (B,), device=motions.device, dtype=torch.long).detach()
        texts = texts[torch.arange(B,device=motions.device, dtype=torch.long), text_idx]

        # bp_code (tgt)
        motions_noise = torch.randn_like(motions)
        motions_t = self.noise_scheduler.add_noise(motions.clone().cpu(), motions_noise.cpu(), t.cpu()).detach().to(motions.device)

        ############
        ###    diffusion step
        ############
        control = self.cmdm_condition_step(t, motions_t, hint, hint_mask, texts, masks, progress)
        motions_0 = self.mdm_diffusion_step(t, motions_t, control, texts, masks, progress)

        ### total loss
        loss = self.loss_rec(motions_0, motions, masks)
        self.log(f"{mode}/loss", loss)
        return loss

    # implemented in eval script
    def test_step(self, train_batch, batch_idx):
        mode = 'test'
        pass
