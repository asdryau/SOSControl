import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from pathlib import Path

from utils.utils import *
import math
from diffusers import DDIMScheduler, DDPMScheduler

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

    def forward(self, x, placeholder):
        # X: (B,T,E)
        x = x + self.pe[:,:x.shape[1], :x.shape[2]]
        return self.dropout(x)
    
class Diffusion(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.num_train_timesteps = 1000
        self.num_inference_timesteps = 1000
        self.save_hyperparameters()

        # model setting
        self.latent_dim = 512

        # GMD emphasis projection
        self.gmd_proj, self.gmd_proj_inv = init_random_projection_rotxz(10)
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, 0.1)

        ### emb
        self.enc_query = nn.Embedding(2, self.latent_dim)

        # self.linear_motin = nn.Linear(269, self.latent_dim)

        # self.linear_motout = nn.Linear(self.latent_dim, 269)

        self.text_embedding = nn.Linear(512, self.latent_dim)

        ###
        #    modules
        ###
        # self.condlinear = nn.Linear(512, self.latent_dim)
        self.scheduler = DDIMScheduler(num_train_timesteps=self.num_train_timesteps, beta_start=0.0001, beta_end=0.02, \
                                       beta_schedule='squaredcos_cap_v2', clip_sample=False, set_alpha_to_one=False, steps_offset=1,\
                                        prediction_type='sample')
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=self.num_train_timesteps, beta_start=0.0001, beta_end=0.02, \
                                       beta_schedule='squaredcos_cap_v2', variance_type='fixed_small', clip_sample=False,\
                                        prediction_type='sample')
        
        self.time_proj = Timesteps(512, True, 0)
        self.time_embedding = TimestepEmbedding(512, self.latent_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.latent_dim, nhead=8, dim_feedforward=1024, dropout=0.1, batch_first=True)
        self.denoiser = nn.TransformerEncoder(encoder_layer, num_layers=8)

    
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

    def denoise(self, time_emb, motions_t, text_emb, len_mask):
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

        out_seq = self.denoiser(src=mot_seq_t, src_key_padding_mask=~mot_mask)
        return out_seq[:,2:]               #(B,4,E)
       
    def diffusion_step(self, t, motions_t, texts, masks, progress):
            # time emb and text emb
        time_emb = self.time_proj(t).type_as(motions_t)
        time_emb = self.time_embedding(time_emb).unsqueeze(1)               #(B,1,E)

        text_emb = self.text_embedding(texts).unsqueeze(1) 

        ### diffusion training
            # expand to latent sine wave
        # motion
        # motions_t = self.linear_motin(motions_t)
        motions_t = self.sequence_pos_encoder(motions_t, progress)

        motions_0 = self.denoise(time_emb, motions_t, text_emb, masks)
        # motions_0 = self.linear_motout(motions_0)
        return motions_0
    
    def training_step(self, train_batch, batch_idx):
        mode = 'train'

        train_batch = [b.detach() for b in train_batch]
        id, motion_rep, texts, masks, progress = train_batch
        B,T = masks.shape

        # motion_rep = self.latent_reparam(motion_codes, progress)

        # motions = torch.matmul(motions, self.gmd_proj.to(motions.device))

        # bp_all = torch.cat([root, spine, leftFoot, rightFoot, leftHand, rightHand], dim=-1)

        # text_emb
        text_idx = torch.randint(0, 12, (B,), device=motion_rep.device, dtype=torch.long).detach()
        texts = texts[torch.arange(B,device=motion_rep.device, dtype=torch.long), text_idx]

        # timestep
        t = torch.randint(0, self.num_train_timesteps, (B, ), device=motion_rep.device, dtype=torch.long).detach()

        # bp_code (tgt)
        motion_rep_noise = torch.randn_like(motion_rep)
        motion_rep_t = self.noise_scheduler.add_noise(motion_rep.clone().cpu(), motion_rep_noise.cpu(), t.cpu()).detach().to(motion_rep.device)

        ############
        ###    diffusion step
        ############
        motion_rep_0 = self.diffusion_step(t, motion_rep_t, texts, masks, progress)

        ### total loss
        loss = self.loss_rec(motion_rep_0, motion_rep, masks)
        self.log(f"{mode}/loss", loss)
        return loss

    def validation_step(self, train_batch, batch_idx):
        mode = 'valid'

        train_batch = [b.detach() for b in train_batch]
        id, motion_rep, texts, masks, progress = train_batch
        B,T = masks.shape

        # motion_rep = self.latent_reparam(motion_codes, progress)

        # motions = torch.matmul(motions, self.gmd_proj.to(motions.device))

        # bp_all = torch.cat([root, spine, leftFoot, rightFoot, leftHand, rightHand], dim=-1)

        # text_emb
        text_idx = torch.randint(0, 12, (B,), device=motion_rep.device, dtype=torch.long).detach()
        texts = texts[torch.arange(B,device=motion_rep.device, dtype=torch.long), text_idx]
        
        # timestep
        t = torch.randint(0, self.num_train_timesteps, (B, ), device=motion_rep.device, dtype=torch.long).detach()

        # bp_code (tgt)
        motion_rep_noise = torch.randn_like(motion_rep)
        motion_rep_t = self.noise_scheduler.add_noise(motion_rep.clone().cpu(), motion_rep_noise.cpu(), t.cpu()).detach().to(motion_rep.device)

        ############
        ###    diffusion step
        ############
        motion_rep_0 = self.diffusion_step(t, motion_rep_t, texts, masks, progress)

        ### total loss
        loss = self.loss_rec(motion_rep_0, motion_rep, masks)
        self.log(f"{mode}/loss", loss)
        return loss

    # implemented in eval script
    def test_step(self, train_batch, batch_idx):
        mode = 'test'
        pass
