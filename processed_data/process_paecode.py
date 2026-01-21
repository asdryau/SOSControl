import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import pickle
from pathlib import Path

from utils.utils import *

import matplotlib
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt


if __name__ == "__main__":


    with open(f"./processed_data/posetraj_data.pkl", 'rb') as handle:
        motion_dict = pickle.load(handle)

    ###############################################
    ### motion latent
    from base_models.MotionPAE_proj10.model import MotionPAE
    model = MotionPAE.load_from_checkpoint(f"./base_models/MotionPAE_proj10/lightning_logs/version_0/checkpoints/last.ckpt")

    motion_code_dict = {}
    for k in motion_dict:
        mot_k = motion_dict[k]
        mot_k = mot_k[:196]
        mask_k = torch.ones((mot_k.shape[0]), device=mot_k.device, dtype=torch.bool).unsqueeze(0)
        progress_k = torch.linspace(-1,1,mot_k.shape[0]).unsqueeze(0)

        mot_k = torch.matmul(mot_k, model.gmd_proj.to(mot_k.device)).unsqueeze(0)
        code_k = model.encode(mot_k, mask_k, progress_k)
        motion_code_dict[k] = code_k.detach().clone().requires_grad_(False)
        print(k, mot_k.shape, code_k.shape, mot_k.dtype, code_k.dtype)
    
    # save
    with open(f"./processed_data/MotionPAE_latent10.pkl", 'wb') as handle:
        pickle.dump(motion_code_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)