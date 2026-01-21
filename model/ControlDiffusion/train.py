import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import pickle
from pathlib import Path

from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from pathlib import Path
#get current foldername
current_folder = str(Path(__file__).parent)
current_folder_name = str(Path(__file__).parent.stem)
# append current_folder tp system_path
import sys
sys.path.append(current_folder)
from .diffusion import ControlDiffusion
from .datamodule import DataModule

if __name__ == "__main__":
    batch_size = 256
    logger = TensorBoardLogger(f"{current_folder_name}")

    dm = DataModule(batch_size=batch_size)
    
    # initialize model
    mdm_path = './model/Diffusion/lightning_logs/version_0/checkpoints/last.ckpt'
    model = ControlDiffusion(mdm_path)

    # # model checkpoint
    checkpoint_callback = ModelCheckpoint(save_last=True, monitor="valid/loss", every_n_train_steps=1000)

    # # Train
    extra_trainer_args = {"precision":16}
    if torch.cuda.is_available():# and not debug:
        extra_trainer_args["gpus"] = -1
        extra_trainer_args["strategy"] = "dp"
        print("cuda available! use all gpu in the machine")

    max_epochs = 300000 # 2000 per run is fine
    check_val_every_n_epoch = 1


    trainer = pl.Trainer(max_epochs=max_epochs, logger=logger, check_val_every_n_epoch=check_val_every_n_epoch, callbacks=[checkpoint_callback], **extra_trainer_args)  #gradient_clip_val=0.5,
    trainer.fit(model, dm)
