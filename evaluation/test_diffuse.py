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

from utils.process_LP import motion_to_contLP, all_directions

cuda_device = 0
torch.cuda.set_device(cuda_device)

# process range in test dataset (4198)
start_process_idx, end_process_idx = 0, -1
batch_size = 16


experiment_list = [
    {"salient_thresh":0.9, "diff_prop":1, "diff_weight":300, "softmax_scale":1},
]

### load test dtatmodule
from evaluation.test_datamodule import DataModule
dm = DataModule(batch_size=1)
dm.setup()
test_data = dm.test_dataloader()
processed_data = [test_batch for test_batch in iter(test_data)]


for exp in experiment_list:
    salient_thresh = exp["salient_thresh"]
    diff_prop = exp["diff_prop"]
    diff_weight = exp["diff_weight"]
    softmax_scale = exp["softmax_scale"]

    salient_thresh_str = "{:.2f}".format(salient_thresh).replace(".", "")
    output_path = Path(f"./evaluation/temp_repo/rep-sal_{salient_thresh_str}-diff_{diff_prop}-w_{diff_weight}-softmax_{softmax_scale}-contPAE")
    output_path.mkdir(parents=True, exist_ok=True)


    ### load models
    from model.ControlDiffusion.diffusion import ControlDiffusion
    mdm_path = './model/Diffusion/lightning_logs/version_0/checkpoints/last.ckpt'
    motion_model = ControlDiffusion.load_from_checkpoint(f"./model/ControlDiffusion/lightning_logs/version_0/checkpoints/last.ckpt", mdm_path=mdm_path)
    motion_model = motion_model.eval().cuda()

    from model.ControlPAE.model import ControlMotionPAE
    pae_path = './model/PAE/lightning_logs/version_0/checkpoints/last.ckpt'
    controlmdm_path = './model/ControlDiffusion/lightning_logs/version_0/checkpoints/last.ckpt'
    model_motionPAE = ControlMotionPAE.load_from_checkpoint(f"./model/ControlPAE/lightning_logs/version_0/checkpoints/last.ckpt", \
                                                            controlmdm_path=controlmdm_path, mdm_path=mdm_path, pae_path=pae_path)
    model_motionPAE = model_motionPAE.eval().cuda()

    rep_mean = torch.from_numpy(np.load(f"./model/rep10_mean.npy"))
    rep_std = torch.from_numpy(np.load(f"./model/rep10_std.npy"))

    ################################
    def transform(data, mean, std):
        data = (data - mean.to(data.device)) / std.to(data.device)
        return data

    def inv_transform(data, mean, std):
        data = data * std.to(data.device) + mean.to(data.device)
        return data

    def latent_reparam(latents, progress):
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

    def LP_diffuse(hint, hint_mask, masks, texts, progress):
        # discLP: , m_len: , texts:
        ##########
        #   motion encoding
        ##########
        B,T = hint.shape[:2]

        hint = hint.cuda()
        hint_mask = hint_mask.to(hint.device)
        masks = masks.to(hint.device)
        progress = progress.to(hint.device)
        texts = texts.to(hint.device)

        input_discLP = hint.reshape((B,-1,6,3))
        hint_mask_clean = hint_mask.reshape((B,T,6,3))[:,:,:,0:1]

        motion_model.scheduler.set_timesteps(99)
        # motion_model.code_scheduler.set_timesteps(99)
        timesteps = motion_model.scheduler.timesteps.to(hint.device)

        # motion_code_pred = torch.randn((B,4,512)).to(discLP.device) * motion_model.code_scheduler.init_noise_sigma
        motion_pred = torch.randn((B,T,512)).to(hint.device) * motion_model.scheduler.init_noise_sigma

        ####################################################
        for i in timesteps:
            t = torch.tensor([i], device=motion_pred.device, dtype=torch.long)
            with torch.set_grad_enabled(True):
                motion_pred.requires_grad_(True)
                # motion_code_pred.requires_grad_(True)
                control = motion_model.cmdm_condition_step(t, motion_pred, \
                                                        hint, hint_mask, texts, masks, None)
                motion_rep0 = motion_model.mdm_diffusion_step(t, motion_pred, control, texts, masks, None)

                if i <= 1:
                    continue

                for _ in range(diff_prop):
                    #################
                    # diffusion time optimization
                    #################
                    motion_0 = inv_transform(motion_rep0, rep_mean, rep_std)
                    control = model_motionPAE.c_decode(motion_0, hint, hint_mask, masks, progress)
                    rec_motion = model_motionPAE.decode(motion_0, control, masks, progress)
                    rec_motion = torch.matmul(rec_motion, model_motionPAE.gmd_proj_inv.to(motion_0.device))

                    # fork loop
                    rec_contLP_program = []
                    for b in range(B):
                        rec_contLP_program.append(torch.jit.fork(motion_to_contLP, rec_motion[b])) 
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

                    ####
                    discLP_diff_scaled = discLP_diff * diff_weight
                    discLP_grad_gmd = torch.autograd.grad([discLP_diff_scaled,],[motion_rep0],create_graph=True)[0] * masks.float().unsqueeze(-1)
                    motion_rep0 = motion_rep0 - discLP_grad_gmd.detach()

                if diff_prop > 0 and i % 100 == 1:
                    print("optimize:", i, discLP_diff)

            motion_pred = motion_model.scheduler.step(motion_rep0, i, motion_pred).prev_sample
            motion_pred = motion_pred.detach()

        motion_rep0 = motion_rep0.detach().cpu()
        return motion_rep0

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

        # test saliency extracted SOS script
        hint_weight_maxbp = torch.amax(hint_weight, dim=1)                              #(B,6)
        hint_weight_maxglobal = torch.amax(hint_weight_maxbp, dim=1).unsqueeze(-1)      #(B,1)
        hint_weight = hint_weight / hint_weight_maxglobal.unsqueeze(1)
        hint_mask = (hint_weight >= salient_thresh)                                     #(B,T,6)
        
        # plt.imshow(hint_mask[0])
        # plt.show()
        # continue

        # order mask hint   (order in (B,T,5))
        hint_mask = hint_mask.unsqueeze(-1).expand((-1,-1,-1,3))
        hint_mask = hint_mask.reshape((B,T,18)).detach()

        hint = hint.reshape((B,T,18)).detach()
        hint = hint * hint_mask

        mask = mask[:,:hint.shape[1]]
        progress = progress[:,:hint.shape[1]]
        rec_motion_rep = LP_diffuse(hint, hint_mask, mask, texts, progress)

        ################

        # convert back to smpl
        for b in range(rec_motion_rep.shape[0]):
            id_b = int(id[b].detach().cpu().numpy())
            rec_motion_rep_b = rec_motion_rep[b]
            m_len = mask[b].sum()
            rec_motion_rep_b = rec_motion_rep_b[:m_len]
            # print(rec_motion_rep_b.shape)
            np.save(output_path/f"{id_b}.npy", rec_motion_rep_b)

        print(f"{id}, --- {time.time() - start_time} seconds ---")
