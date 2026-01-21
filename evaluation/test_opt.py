import numpy as np
import torch
import time
from utils.rotation_conversion import *
from utils.utils import *
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence

import matplotlib
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt

from utils.process_LP import motion_to_contLP, motion_to_smpl, spatial_LP_discretization, all_directions

cuda_device = 0
torch.cuda.set_device(cuda_device)

# process range in test dataset (4198)

start_process_idx, end_process_idx = 0, -1
batch_size = 16

# inference pipeline parameter

fc_vel_thresh = 0.50
fc_height_thresh = 0.05

input_motion_folders = [
    ["rep-sal_090-diff_1-w_300-softmax_1-contPAE", 100, 3000],
]

### load test datamodule
from evaluation.test_datamodule import DataModule
dm = DataModule(batch_size=1)
dm.setup()
test_data = dm.test_dataloader()
processed_data = [test_batch for test_batch in iter(test_data)]

for p_list in input_motion_folders:

    input_path = p_list[0]
    num_loop = p_list[1]
    opt_weight = p_list[2]

    setting = input_path.split("-")[1].split("_")
    test_saliency = True
    test_global = True
    salient_thresh = int(setting[1]) / 100

    print("====================")
    print(input_path)
    print(f"salient_thresh:{salient_thresh}")


    softmax_scale = float(input_path.split("-softmax_")[1].split("-")[0])

    input_path = Path(f"./evaluation/temp_repo") / input_path

    output_path = Path(f"./evaluation/generated_repo") / (str(input_path.stem) + f"-opt_{num_loop}-o_{opt_weight}-contPAE")
    output_path.mkdir(parents=True, exist_ok=True)

    ### load diffusion
    start_time = time.time()

    from model.ControlPAE.model import ControlMotionPAE
    pae_path = './model/PAE/lightning_logs/version_0/checkpoints/last.ckpt'
    mdm_path = './model/Diffusion/lightning_logs/version_0/checkpoints/last.ckpt'
    controlmdm_path = './model/ControlDiffusion/lightning_logs/version_0/checkpoints/last.ckpt'
    model_motionPAE = ControlMotionPAE.load_from_checkpoint(f"./model/ControlPAE/lightning_logs/version_0/checkpoints/last.ckpt", \
                                                            controlmdm_path=controlmdm_path, mdm_path=mdm_path, pae_path=pae_path)
    model_motionPAE = model_motionPAE.eval().cuda()

    rep_mean = torch.from_numpy(np.load(f"./model/rep10_mean.npy"))
    rep_std = torch.from_numpy(np.load(f"./model/rep10_std.npy"))
    print(f"load model time, --- {time.time() - start_time} seconds ---")

    ################################
    def transform(data, mean, std):
        data = (data - mean.to(data.device)) / std.to(data.device)
        return data

    def inv_transform(data, mean, std):
        data = data * std.to(data.device) + mean.to(data.device)
        return data

    def optimize_motion_rep0(motion_rep0, masks, progress, hint, hint_mask, num_loop=0):
        B,T = motion_rep0.shape[:2]
        input_discLP = hint.reshape((B,-1,6,3))[:,:]
        hint_mask_clean = hint_mask.reshape((B,T,6,3))[:,:,:,0:1]
        for l in range(num_loop):
            with torch.set_grad_enabled(True):
                motion_rep0.requires_grad_(True)
                motion_0 = inv_transform(motion_rep0, rep_mean, rep_std)
                ### motion guidance
                # rec_motion = model_motionPAE.decode(motion_0, masks, progress)
                control = model_motionPAE.c_decode(motion_0, hint, hint_mask, masks, progress)
                rec_motion = model_motionPAE.decode(motion_0, control, masks, progress)
                rec_motion = torch.matmul(rec_motion, model_motionPAE.gmd_proj_inv.to(motion_0.device))

                # fork loop
                rec_contLP_program = []
                for b in range(B):
                    rec_contLP_program.append(torch.jit.fork(motion_to_contLP, rec_motion[b])) #LP_weight[b][:len_b]
                # wait loop
                rec_contLP = []
                for b in range(B):
                    out_b = torch.jit.wait(rec_contLP_program[b])
                    rec_contLP.append(out_b)

                ###
                rec_contLP = torch.stack(rec_contLP, dim=0).reshape((B,-1,6,3))       #  (B,T,6,3)
                ### calc motion_GT_discLP_diff
                rec_contLP_norm = torch.norm(rec_contLP, p=2, dim=-1, keepdim=True)
                rec_contLP_valid_mask = (rec_contLP_norm > 1e-5).float()
                rec_contLP_norm = rec_contLP_norm * rec_contLP_valid_mask + 1. * (1-rec_contLP_valid_mask)
                rec_contLP = rec_contLP / rec_contLP_norm  # make it unit length in case

                    # newLPfn
                rec_contLP_dir = torch.nn.functional.softmax(rec_contLP @ all_directions.t().to(motion_rep0.device) * softmax_scale, dim=-1)   #(T,5,26)
                discLP_dir = torch.nn.functional.softmax(input_discLP @ all_directions.t().to(motion_rep0.device) * softmax_scale, dim=-1)            #(T,5,26)
                discLP_diff = (discLP_dir - rec_contLP_dir).abs() * hint_mask_clean
                discLP_diff = discLP_diff.sum() / hint_mask_clean.reshape((-1)).float().sum()

                if l % 10 == 0:
                    print("optimize:", discLP_diff)
                discLP_diff = discLP_diff * opt_weight
                discLP_grad_gmd = torch.autograd.grad([discLP_diff,],[motion_rep0],create_graph=True)[0] * masks.float().unsqueeze(-1)
            motion_rep0 = motion_rep0 - discLP_grad_gmd
            motion_rep0 = motion_rep0.detach()

        return motion_rep0

    def optimization(motion_rep0, hint, hint_mask, masks, texts, progress):
        # discLP: , m_len: , texts:
        ##########
        #   motion encoding
        ##########

        hint = hint.cuda()
        masks = masks.to(hint.device)
        progress = progress.to(hint.device)
        texts = texts.to(hint.device)
        motion_rep0 = motion_rep0.to(hint.device)
        hint_mask = hint_mask.to(hint.device)

        motion_rep0 = optimize_motion_rep0(motion_rep0.detach(), masks, progress, hint, hint_mask, num_loop=num_loop)

        motion_0 = inv_transform(motion_rep0, rep_mean, rep_std)
        # rec_motion = model_motionPAE.decode(motion_0, masks, progress)
        control = model_motionPAE.c_decode(motion_0, hint, hint_mask, masks, progress)
        rec_motion = model_motionPAE.decode(motion_0, control, masks, progress)
        rec_motion = torch.matmul(rec_motion, model_motionPAE.gmd_proj_inv.to(motion_0.device))
        rec_motion = rec_motion.detach().cpu()
        return rec_motion

    ########################################################################################################
    ### eval script

    start_time = time.time()
    end_process_idx = end_process_idx if end_process_idx >= 0 else len(processed_data)
    for test_idx in range(start_process_idx, end_process_idx, batch_size):
        test_batch = processed_data[test_idx:test_idx+batch_size]
        test_batch = zip(*test_batch)
        test_batch = [torch.cat(d, dim=0) for d in test_batch]
        id, sample_motion, contLP, discLP, LP_weight, texts, mask, progress = test_batch

        # print(sample_motion.shape)
        B,T = sample_motion.shape[:2]
        hint_weight = LP_weight
        hint = discLP

        hint_weight_maxbp = torch.amax(hint_weight, dim=1)                              #(B,6)
        hint_weight_maxglobal = torch.amax(hint_weight_maxbp, dim=1).unsqueeze(-1)      #(B,1)
        if test_global:
            hint_weight = hint_weight / hint_weight_maxglobal.unsqueeze(1)
        else:
            hint_weight = hint_weight / hint_weight_maxbp.unsqueeze(1)
        hint_mask = (hint_weight >= salient_thresh)                                     #(B,T,6)
        
        # plt.imshow(hint_mask[0])
        # plt.show()
        # exit()
        # print(hint_mask.sum(), hint_mask.shape)
        # order mask hint   (order in (B,T,5))
        hint_mask = hint_mask.unsqueeze(-1).expand((-1,-1,-1,3))
        hint_mask = hint_mask.reshape((B,T,18)).detach()

        hint = hint.reshape((B,T,18)).detach()


        hint = hint.detach().cuda()
        mask = mask[:,:contLP.shape[1]]
        progress = progress[:,:contLP.shape[1]]


        motion_reps = []
        for b in range(B):
            id_b = int(id[b].detach().cpu().numpy())
            data = np.load(input_path/f"{id_b}.npy")
            motion_reps.append(torch.from_numpy(data).detach().cuda())


        motion_reps = pad_sequence(motion_reps, batch_first=True)
        hint = hint[:,:motion_reps.shape[1]]
        hint_mask = hint_mask[:,:motion_reps.shape[1]]
        mask = mask[:,:motion_reps.shape[1]]
        progress = progress[:,:motion_reps.shape[1]]
        print(motion_reps.shape, hint.shape, hint_mask.shape, mask.shape, texts.shape, progress.shape)
        rec_motion = optimization(motion_reps, hint, hint_mask, mask, texts, progress)

        ################

        # convert back to smpl
        for b in range(rec_motion.shape[0]):
            id_b = int(id[b].detach().cpu().numpy())
            rec_motion_b = rec_motion[b]
            m_len = mask[b].sum()
            traj, mot = motion_to_smpl(rec_motion_b)
            mot = mot[:m_len]
            traj = traj[:m_len]
            # print(id_b, m_len)
            save_motion(output_path/f"{id_b}_mot.npz", traj, mot.reshape((-1,22*3)))

        print(f"{id}, --- {time.time() - start_time} seconds ---")
