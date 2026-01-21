import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import pickle
import csv
import time
from pathlib import Path

from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from scipy.ndimage import uniform_filter1d

from utils.utils import *
from utils.process_LP import *

import matplotlib
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

eval_motion_folders = [
    "rep-sal_090-diff_1-w_300-softmax_1-contPAE-opt_100-o_3000-contPAE",
]
start_process_idx, end_process_idx = 0, -1

########################################################################################################
# reference: https://github.com/cr7anand/neural_temporal_models/blob/master/metrics.py
def compute_npss(euler_gt_sequences, euler_pred_sequences):

    # computing 1) fourier coeffs 2)power of fft 3) normalizing power of fft dim-wise 4) cumsum over freq. 5) EMD
    gt_fourier_coeffs = np.zeros(euler_gt_sequences.shape)
    pred_fourier_coeffs = np.zeros(euler_pred_sequences.shape)

    # power vars
    gt_power = np.zeros((gt_fourier_coeffs.shape))
    pred_power = np.zeros((gt_fourier_coeffs.shape))

    # normalizing power vars
    gt_norm_power = np.zeros(gt_fourier_coeffs.shape)
    pred_norm_power = np.zeros(gt_fourier_coeffs.shape)

    cdf_gt_power = np.zeros(gt_norm_power.shape)
    cdf_pred_power = np.zeros(pred_norm_power.shape)

    emd = np.zeros(cdf_pred_power.shape[0:3:2])

    # used to store powers of feature_dims and sequences used for avg later
    seq_feature_power = np.zeros(euler_gt_sequences.shape[0:3:2])
    power_weighted_emd = 0

    for s in range(euler_gt_sequences.shape[0]):

        for d in range(euler_gt_sequences.shape[2]):
            gt_fourier_coeffs[s,:,d] = np.fft.fft(euler_gt_sequences[s,:,d]) # slice is 1D array
            pred_fourier_coeffs[s,:,d] = np.fft.fft(euler_pred_sequences[s,:,d])

            # computing power of fft per sequence per dim
            gt_power[s,:,d] = np.square(np.absolute(gt_fourier_coeffs[s,:,d]))
            pred_power[s,:,d] = np.square(np.absolute(pred_fourier_coeffs[s,:,d]))

            # matching power of gt and pred sequences
            gt_total_power = np.sum(gt_power[s,:,d])
            pred_total_power = np.sum(pred_power[s,:,d])
            #power_diff = gt_total_power - pred_total_power

            # adding power diff to zero freq of pred seq
            #pred_power[s,0,d] = pred_power[s,0,d] + power_diff

            # computing seq_power and feature_dims power
            seq_feature_power[s,d] = gt_total_power

            # normalizing power per sequence per dim
            if gt_total_power != 0:
                gt_norm_power[s,:,d] = gt_power[s,:,d] / gt_total_power

            if pred_total_power !=0:
                pred_norm_power[s,:,d] = pred_power[s,:,d] / pred_total_power

            # computing cumsum over freq
            cdf_gt_power[s,:,d] = np.cumsum(gt_norm_power[s,:,d]) # slice is 1D
            cdf_pred_power[s,:,d] = np.cumsum(pred_norm_power[s,:,d])

            # computing EMD
            emd[s,d] = np.linalg.norm((cdf_pred_power[s,:,d] - cdf_gt_power[s,:,d]), ord=1)

    # computing weighted emd (by sequence and feature powers)
    power_weighted_emd = np.average(emd, weights=seq_feature_power)

    return power_weighted_emd

def calculate_skating_ratio(motions):
    thresh_height = 0.05 # 10
    fps = 20.0
    thresh_vel = 0.50 # 20 cm /s
    avg_window = 5 # frames

    motions = motions.unsqueeze(0)
    motions = motions.permute((0,2,3,1))
    batch_size = motions.shape[0]
    # 10 left, 11 right foot. XY plane, z up
    # motions [bs, 22, 3, max_len]
    verts_feet = motions[:, [10, 11], :, :].detach().cpu().numpy()  # [bs, 2, 3, max_len]
    verts_feet_plane_vel = np.linalg.norm(verts_feet[:, :, [0, 1], 1:] - verts_feet[:, :, [0, 1], :-1],  axis=2) * fps  # [bs, 2, max_len-1]
    # [bs, 2, max_len-1]
    vel_avg = uniform_filter1d(verts_feet_plane_vel, axis=-1, size=avg_window, mode='constant', origin=0)

    verts_feet_height = verts_feet[:, :, 2, :]  # [bs, 2, max_len]
    # If feet touch ground in agjecent frames
    feet_contact = np.logical_and((verts_feet_height[:, :, :-1] < thresh_height), (verts_feet_height[:, :, 1:] < thresh_height))  # [bs, 2, max_len - 1]
    # skate velocity
    skate_vel = feet_contact * vel_avg

    # it must both skating in the current frame
    skating = np.logical_and(feet_contact, (verts_feet_plane_vel > thresh_vel))
    # and also skate in the windows of frames
    skating = np.logical_and(skating, (vel_avg > thresh_vel))

    # Both feet slide
    skating = np.logical_or(skating[:, 0, :], skating[:, 1, :]) # [bs, max_len -1]
    skating_ratio = np.sum(skating, axis=1) / skating.shape[1]

    return skating_ratio, skate_vel


for eval_motion_folder in eval_motion_folders:
# eval_motion_folder = Path(f"./evaluation/results/controlnet_rep10-tok3-100-fc_True-wDiff_False")

    setting = eval_motion_folder.split("-")[1].split("_")
    salient_thresh = int(setting[1]) / 100
        

    # eval_motion_folder = Path(f"./evaluation/reference")
    eval_motion_folder = Path(f"./evaluation/generated_repo") / eval_motion_folder
    eval_motion_path = eval_motion_folder.rglob("*.npz")
    eval_motion_path = [m for m in eval_motion_path]
    eval_motion_ids = [int(str(m.stem).split("_")[0]) for m in eval_motion_path]

    print("====================")
    print(eval_motion_folder)
    print(f"salient_thresh:{salient_thresh}")

    ### load test dtatmodule
    from evaluation.test_datamodule import DataModule, motion_preprocess
    dm = DataModule(batch_size=1)
    dm.setup()
    test_data = dm.test_dataloader()
    processed_data = [test_batch for test_batch in iter(test_data)]

    start_time = time.time()

    test_batch = zip(*processed_data)
    test_batch = [torch.cat(d, dim=0) for d in test_batch]
    id, sample_motion, contLP, discLP, LP_weight, texts, mask, progress = test_batch

    B = sample_motion.shape[0]

    # metrics
    npss6d_all = []
    l26d_all = []
    l2p_all = []
    l2q_all = []
    discLP_acc_all = []
    LP_dotdiff_all = []
    fc_all = []
    fsr_all = []
    rotjerk_all = []

    for b in range(B):
        id_b = int(id[b].detach().cpu().numpy())
        len_b = mask[b].sum()-1     # discard 1 more frame (velocity calculation)

        #####
        #   get associated eval motion
        #####
        try:
            index = eval_motion_ids.index(id_b)
            mot = np.load(eval_motion_path[index])
            gen_poses = torch.from_numpy(mot['poses']).cuda()
            gen_trans = torch.from_numpy(mot['trans']).cuda()
            gen_poses = gen_poses[:len_b,:66].reshape((-1,22,3))
            gen_trans = gen_trans[:len_b]
        except ValueError:
            continue

        #####
        #   tidy up eval information
        #####
        hint_weight = LP_weight[b:b+1].cuda()
        hint = discLP[b:b+1].cuda()
        B,T = hint.shape[:2]

        # 
        hint_weight_maxbp = torch.amax(hint_weight, dim=1)                              #(B,6)
        hint_weight_maxglobal = torch.amax(hint_weight_maxbp, dim=1).unsqueeze(-1)      #(B,1)
        hint_weight = hint_weight / hint_weight_maxglobal.unsqueeze(1)
        hint_mask = (hint_weight >= salient_thresh)                                     #(B,T,6)

        # order mask hint   (order in (B,T,5))
        hint_mask_pad = hint_mask.unsqueeze(-1).expand((-1,-1,-1,3))
        hint_mask_pad = hint_mask_pad.reshape((B,T,18)).detach()
        hint = hint.reshape((B,T,18)).detach()
        discLP_b = hint * hint_mask_pad
        discLP_b = discLP_b[0,:len_b]

        len_b = min(len_b, gen_poses.shape[0])

        sample_trans, sample_poses = motion_to_smpl(sample_motion[b][:len_b])
        sample_poses = sample_poses[:,:66].reshape((-1,22,3)).cuda()
        sample_trans = sample_trans.cuda()

        ###
        #   evaluation metrics
        ###

        ### REC (L26D / NPSS / L2P)

        global_positions = forward_kinematics(sample_trans, sample_poses) #(T,J,3)
        generated_positions = forward_kinematics(gen_trans, gen_poses) #(T,J,3)
        l2p = generated_positions - global_positions       #(T,21)
        l2p = torch.mean(torch.norm(l2p, dim=2)).unsqueeze(0)                 #(1,)
        l2p_all.append(l2p.detach().cpu())

        l2q = torch.norm(axis_angle_to_quaternion(gen_poses) - axis_angle_to_quaternion(sample_poses), dim=2)       #(T,21)
        l2q = torch.mean(l2q).unsqueeze(0)                 #(1,)
        l2q_all.append(l2q.detach().cpu())

        l26d = torch.norm(matrix_to_rotation_6d(axis_angle_to_matrix(gen_poses)) - matrix_to_rotation_6d(axis_angle_to_matrix(sample_poses)), dim=2)       #(T,21)
        l26d = torch.mean(l26d).unsqueeze(0)                 #(1,)
        l26d_all.append(l26d.detach().cpu())

        poses_6d = matrix_to_rotation_6d(axis_angle_to_matrix(sample_poses)).reshape((-1,22*6)).unsqueeze(0)       #(1,T,21*3)
        generated_6d = matrix_to_rotation_6d(axis_angle_to_matrix(gen_poses)).reshape((-1,22*6)).unsqueeze(0)
        npss6d = compute_npss(poses_6d.detach().cpu().numpy(), generated_6d.detach().cpu().numpy())
        npss6d_all.append(npss6d)

        ### other (discLP / fc)
        hint_mask = hint_mask[0, :gen_poses.shape[0]].float().unsqueeze(-1)

        discLP_b = discLP_b[:,:18].reshape((-1,6,3))
        discLP_norm = torch.norm(discLP_b, p=2, dim=-1, keepdim=True)
        # discLP_valid_mask = (discLP_norm > 1e-5).float()
        discLP_norm = discLP_norm * hint_mask + 1. * (1-hint_mask)
        discLP_b = discLP_b / discLP_norm  # make it unit length in case
        discLP_norm_argmax = torch.argmax(discLP_b.float() @ all_directions.t().to(sample_poses.device), dim=-1)
        discLP_norm_onehot = torch.nn.functional.one_hot(discLP_norm_argmax, num_classes=all_directions.shape[0])

        gen_LP = posetraj_to_contLP(gen_poses.reshape((-1,66))).reshape((-1,6,3))
        gen_LP_norm = torch.norm(gen_LP, p=2, dim=-1, keepdim=True)
        # gen_LP_valid_mask = (gen_LP_norm > 1e-5).float()
        gen_LP_norm = gen_LP_norm * hint_mask + 1. * (1-hint_mask)
        gen_LP = gen_LP / gen_LP_norm  # make it unit length in case
        gen_LP_norm_argmax = torch.argmax(gen_LP @ all_directions.t().float().to(gen_poses.device), dim=-1)
        gen_LP_norm_onehot = torch.nn.functional.one_hot(gen_LP_norm_argmax, num_classes=all_directions.shape[0])

        # print(discLP_valid_mask.shape, gen_LP_norm_onehot.shape, discLP_norm_onehot.shape, discLP_norm.shape, gen_LP_norm.shape)

        discLP_diff = (gen_LP_norm_onehot - discLP_norm_onehot).abs() * hint_mask   #  (B,T,5)
        discLP_diff = discLP_diff.sum() / hint_mask.reshape((-1)).float().sum()
        discLP_acc = (1 - discLP_diff / 2)

        LP_dotdiff = (1 - torch.sum(discLP_norm * gen_LP_norm, dim=-1)) * hint_mask.squeeze(-1)   #  (B,T,5)
        LP_dotdiff = LP_dotdiff.sum() / hint_mask.reshape((-1)).float().sum()
        discLP_acc_all.append(discLP_acc.detach().cpu().numpy())
        LP_dotdiff_all.append(LP_dotdiff.detach().cpu().numpy())

        ### fc
        feet_l_v = torch.norm(generated_positions[1:, 10] - generated_positions[:-1, 10], dim=-1)
        feet_r_v = torch.norm(generated_positions[1:, 11] - generated_positions[:-1, 11], dim=-1)
        feet_l_h = (generated_positions[:-1, 10, 2] > 0.002).float()
        feet_r_h = (generated_positions[:-1, 11, 2] > 0.002).float()
        fc = torch.sum(feet_l_v * feet_l_h) + torch.sum(feet_r_h * feet_r_v)
        fc = fc / mask.reshape((-1)).float().sum()
        fc_all.append(fc.detach().cpu().numpy())

        fsr, _ = calculate_skating_ratio(generated_positions)
        fsr_all.append(fsr)

        ### smoothness
        gen_poses = axis_angle_to_quaternion(gen_poses)
        generated_rotvel = quaternion_multiply(gen_poses[1:], quaternion_invert(gen_poses[:-1]))
        generated_rotacc = quaternion_multiply(generated_rotvel[1:], quaternion_invert(generated_rotvel[:-1]))
        generated_rotjerk = quaternion_multiply(generated_rotacc[1:], quaternion_invert(generated_rotacc[:-1]))
        generated_rotjerk = quaternion_to_axis_angle(generated_rotjerk)
        generated_rotjerk = torch.sqrt(torch.mean(torch.square(generated_rotjerk))).unsqueeze(0)                   #(1,)
        rotjerk_all.append(generated_rotjerk.detach().cpu().numpy())


    l2p_all = torch.cat(l2p_all, dim=0).mean()
    l2q_all = torch.cat(l2q_all, dim=0).mean()
    l26d_all = torch.cat(l26d_all, dim=0).mean()
    npss6d_all = np.mean(npss6d_all)
    discLP_acc_all = np.mean(discLP_acc_all)
    LP_dotdiff_all = np.mean(LP_dotdiff_all)
    fc_all = np.mean(fc_all)
    fsr_all = np.mean(fsr_all)
    rotjerk_all = np.mean(rotjerk_all)

    metrics = {}
    metrics["L2P"] = l2p_all.detach().cpu().numpy()
    metrics["L2Q"] = l2q_all.detach().cpu().numpy()
    metrics["L26D"] = l26d_all.detach().cpu().numpy()
    metrics["NPSS_6D"] = npss6d_all
    metrics["DiscLP_Acc"] = discLP_acc_all
    metrics["ContLP_Dot"] = LP_dotdiff_all
    metrics["Foot_Contact"] = fc_all
    metrics["Foot_Skating"] = fsr_all
    metrics["RMS_Jerk"] = rotjerk_all

    savefilename = str(eval_motion_folder) + "_evalresult.csv"
    with open(savefilename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(list(metrics.keys()))
        writer.writerow([float(v) for v in list(metrics.values())])