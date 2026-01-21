import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from pathlib import Path
from .datamodule import motion_to_smpl

from utils.utils import *
# import clip, json

loss_mse = nn.MSELoss()
loss_l1 = nn.L1Loss()

# reference: https://github.com/korrawe/guided-motion-diffusion/blob/2f6264a9b793333556ef911981983082a1113049/data_loaders/humanml/data/dataset.py
def init_random_projection_rotxz(scale):
    if Path(f"./model/rand_proj_{scale}_rotxz.npy").exists():
        proj_matrix = np.load(f"./model/rand_proj_{scale}_rotxz.npy")
        inv_proj_matrix = np.load(f"./model/inv_rand_proj_{scale}_rotxz.npy")
    else:
        proj_matrix = torch.normal(mean=0, std=1.0, size=(269, 269), dtype=torch.float)  # / np.sqrt(263)
        # scale first 3 values (rot spd, x spd, z spd)
        proj_matrix[[0,7,8], :] *= scale
        proj_matrix = proj_matrix / np.sqrt(269 - 3 + 3 * scale**2)
        inv_proj_matrix = torch.inverse(proj_matrix)

        proj_matrix = proj_matrix.detach().cpu().numpy()
        inv_proj_matrix = inv_proj_matrix.detach().cpu().numpy()
        np.save(f"./model/rand_proj_{scale}_rotxz.npy", proj_matrix)
        np.save(f"./model/inv_rand_proj_{scale}_rotxz.npy", inv_proj_matrix)

    proj_matrix_th = torch.from_numpy(proj_matrix)
    inv_proj_matrix_th = torch.from_numpy(inv_proj_matrix)

    return proj_matrix_th, inv_proj_matrix_th

### reference: MDM
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

    def forward(self, x, placeholder):
        # X: (B,T,E)
        x = x + self.pe[:,:x.shape[1], :x.shape[2]]
        return self.dropout(x)


class MotionPAE(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # GMD emphasis projection
        self.gmd_proj, self.gmd_proj_inv = init_random_projection_rotxz(10)

        # model setting
        self.input_dim = 269
        self.latent_dim = 512


        self.save_hyperparameters()

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, 0.1)


        ###
        #   Encoder
        ###
        self.enc_query = nn.Embedding(4, self.latent_dim)
        self.skelEmbedding = nn.Linear(self.input_dim, self.latent_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.latent_dim, nhead=8, dim_feedforward=1024, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=8)

        ###
        #   Decoder
        ###

        decoder_layer = nn.TransformerEncoderLayer(d_model=self.latent_dim, nhead=8, dim_feedforward=1024, dropout=0.1, batch_first=True)
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=8)
        
        self.finallayer = nn.Linear(self.latent_dim, self.input_dim)


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
    
    def decode(self, reparam_latents, len_mask, progress):
        # (B,T,?), (B,T)          
        # decoder 
        reparam_latents = self.sequence_pos_encoder(reparam_latents, progress)
        output = self.decoder(reparam_latents, src_key_padding_mask=~len_mask)    #(B,T,E)
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
        id, motions, masks, progress = train_batch

        B,T = motions.shape[:2]

        motions = torch.matmul(motions, self.gmd_proj.to(motions.device))

        code = self.encode(motions, masks, progress)
        rep = self.latent_reparam(code, progress)
        rec = self.decode(rep, masks, progress)

        loss = self.loss_rec(rec, motions, masks)

        self.log(f"{mode}/loss", loss)
        return loss


    def validation_step(self, train_batch, batch_idx):
        mode = 'valid'
        train_batch = [b.detach() for b in train_batch]
        id, motions, masks, progress = train_batch

        B,T = motions.shape[:2]

        motions = torch.matmul(motions, self.gmd_proj.to(motions.device))

        code = self.encode(motions, masks, progress)
        rep = self.latent_reparam(code, progress)
        rec = self.decode(rep, masks, progress)

        loss = self.loss_rec(rec, motions, masks)

        self.log(f"{mode}/loss", loss)
        return loss

    def test_step(self, train_batch, batch_idx):
        mode = 'test'
        train_batch = [b.detach() for b in train_batch]
        id, motions, masks, progress = train_batch

        B,T = motions.shape[:2]

        motions = torch.matmul(motions, self.gmd_proj.to(motions.device))

        code = self.encode(motions, masks, progress)
        rep = self.latent_reparam(code, progress)
        rec = self.decode(rep, masks, progress)

        loss = self.loss_rec(rec, motions, masks)

        ######
        #   
        if batch_idx == 0:
            for b in range(rec.shape[0]):
                rec_motion = rec[b]
                rec_motion = torch.matmul(rec_motion, self.gmd_proj_inv.to(rec_motion.device))
                traj, mot = motion_to_smpl(rec_motion)
                mot = mot.reshape(T,22,3)
                l = int(masks[b:b+1].sum())
                mot = mot[:l]
                traj = traj[:l]
                save_motion(f"{b}_rec.npz", traj, mot.reshape((-1,22*3)))

                rec_motion = motions[b]
                rec_motion = torch.matmul(rec_motion, self.gmd_proj_inv.to(rec_motion.device))
                traj, mot = motion_to_smpl(rec_motion)
                mot = mot.reshape(T,22,3)
                l = int(masks[b:b+1].sum())
                mot = mot[:l]
                traj = traj[:l]
                save_motion(f"{b}_mot.npz", traj, mot.reshape((-1,22*3)))

        self.log(f"{mode}/loss", loss)
        return loss
