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

from utils.utils import *

########
#   data preprocessing function
#######
def motion_preprocess(trans, poses):
    # trans : (T,3)
    T = poses.shape[0]
    poses = poses.reshape((T, -1,3)) #(T,J,3)
    # poses = axis_angle_to_quaternion(poses) #(T,J,4)
    
    # print("trans, poses", trans.shape, poses.shape) # torch.Size([47, 3]) torch.Size([47, 24, 4])
    ###########
    # we follow the step from HumanML3D but with some modifications, also we use smplx package and pytorch3D rotation_conversion

    ### 1) FK 
    global_positions = forward_kinematics(trans, poses) #(T,J,3)
    poses = axis_angle_to_quaternion(poses) #(T,J,4)
    ### 2) put on floor (AMASS is Z-up)
    floor_height = global_positions.min(axis=0)[0].min(axis=0)[0][2]
    global_positions[:,:,2] -= floor_height

    # ### 3) set initital frame XY at (0,0)
    global_positions[:,:,0] = global_positions[:,:,0] - global_positions[0:1,0:1,0]      # Note that height (z position) is preserved
    global_positions[:,:,1] = global_positions[:,:,1] - global_positions[0:1,0:1,1]      # Note that height (z position) is preserved

    ### 4) initial frame face Y+
    last_pose = matrix_to_quaternion(torch.eye(3)).to(poses.device)    #(4,)   # define y+ forward orientation
    next_pose = poses[0, 0].clone()               #(4,)   # this is the root orientation of center frame (you can change it to frame0, frameT, etc...)
    root_rotate_start = quaternion_multiply(last_pose, quaternion_invert(next_pose))  # (4,)
        # normalize root_rotate_start to only work on horizontal orientation, not elevation
            ### for AMASS format (Z-up)
    root_rotate_startY = matrix_to_euler_angles(quaternion_to_matrix(root_rotate_start), "XYZ")
    root_rotate_startY[0] = 0.
    root_rotate_startY[1] = 0.
    root_rotate_start = matrix_to_quaternion(euler_angles_to_matrix(root_rotate_startY, "XYZ"))
    poses[:,0] = quaternion_multiply(root_rotate_start.unsqueeze(0), poses[:,0])
    #     ### also rotate global_positions
    global_positions = quaternion_apply(root_rotate_start.unsqueeze(0), global_positions)

    # print("global_positions", global_positions.shape)     #torch.Size([47, 22, 3])

    ### 5) get joint rotation in 6D
    poses_6d = matrix_to_rotation_6d(quaternion_to_matrix(poses))
    # print("poses_6d", poses_6d.shape)   # torch.Size([47, 24, 6])

    ### 6) get root representation !!!!!!
    root_pose = poses[:,0]
    trans = global_positions[:,0]
        # calc root_rotate that set all frame facing Y+
    last_pose = matrix_to_quaternion(torch.eye(3)).to(poses.device)    #(4,)   # define y+ forward orientation
    root_rotate_all = quaternion_multiply(last_pose.unsqueeze(0), quaternion_invert(root_pose))  # (T,4)
    root_rotate_allY = matrix_to_euler_angles(quaternion_to_matrix(root_rotate_all), "XYZ")
    root_rotate_allY[:,0] = 0.
    root_rotate_allY[:,1] = 0.
    root_rotate_all = matrix_to_quaternion(euler_angles_to_matrix(root_rotate_allY, "XYZ"))
        # calc linear velocity
    traj_vel = trans[1:] - trans[:-1]
    traj_vel = quaternion_apply(root_rotate_all[:-1], traj_vel)
        # calc root angular velocity
    root_rotvelZ =  quaternion_multiply(root_pose[1:], quaternion_invert(root_pose[:-1]))
    root_rotvelZ = matrix_to_euler_angles(quaternion_to_matrix(root_rotvelZ), "XYZ")[:,2]
    # print("traj_vel, root_rotvelZ", traj_vel.shape, root_rotvelZ.shape)   # torch.Size([46, 3]) torch.Size([46])

    ### 7) get rifke
    root_rot_Zfiltered = quaternion_multiply(root_rotate_all, root_pose)
        # FK again to obtain rifke
    pose_modified = poses.clone()
    pose_modified[:,0] = root_rot_Zfiltered
    trans_modified = global_positions[:,0].clone()
    trans_modified[:,0] = 0.
    trans_modified[:,1] = 0.
    rifke = forward_kinematics(trans_modified, matrix_to_axis_angle(quaternion_to_matrix(pose_modified))) #(T,J,3)
    # print("root_rot_Zfiltered, rifke", root_rot_Zfiltered.shape, rifke.shape) # torch.Size([47, 4]) torch.Size([47, 22, 3])

    ### 8) get local velocity
    local_vel = global_positions[1:] - global_positions[:-1]
    local_vel = quaternion_apply(root_rotate_all[:-1].unsqueeze(1), local_vel)
    # print("local_vel", local_vel.shape)   # torch.Size([46, 22, 3])

    ### 9) foot contact
    feet_l = (torch.norm(global_positions[1:, [7, 10]] - global_positions[:-1, [7, 10]], dim=2) < 0.002).float()
    feet_r = (torch.norm(global_positions[1:, [8, 11]] - global_positions[:-1, [8, 11]], dim=2) < 0.002).float()
    # print("feet_l, feet_r", feet_l.shape, feet_r.shape)   # torch.Size([46, 2]) torch.Size([46, 2])

    ### concatnate data
    data = torch.cat([root_rotvelZ.unsqueeze(-1), matrix_to_rotation_6d(quaternion_to_matrix(root_rot_Zfiltered[:-1])),\
                        traj_vel[:,0:2], trans[:-1,2:3],\
                        poses_6d[:-1,1:22].reshape(T-1,-1), rifke[:-1,1:22].reshape(T-1,-1), local_vel[:,:22].reshape(T-1,-1), feet_l, feet_r], dim=-1)
    # print("data", data.shape)   #torch.Size([46, 269])
    return data

def motion_to_smpl(motion):
    ### root traj
    root_rotvelZ, root_rot_Zfiltered = motion[:,0], motion[:,1:7]
    traj_vel, transZ = motion[:,7:9], motion[:,9:10]
        # process rotation velocity
    root_rotvelZ = torch.cat([root_rotvelZ[-1:]*0., root_rotvelZ[:-1]], dim=0)  # shift-1 in dim 0
    root_rotZ = torch.cumsum(root_rotvelZ, dim=0)
    root_rotZ = torch.stack([root_rotZ*0., root_rotZ*0., root_rotZ],dim=-1)
    root_rotZ = matrix_to_quaternion(euler_angles_to_matrix(root_rotZ, "XYZ"))

        # reconstruct root_rot
    root_rot_Zfiltered = matrix_to_quaternion(rotation_6d_to_matrix(root_rot_Zfiltered))
    root_rot = quaternion_multiply(root_rotZ, root_rot_Zfiltered)   #(T,4)
    root_rot = matrix_to_axis_angle(quaternion_to_matrix(root_rot))
        # traj
    traj_vel = torch.cat([traj_vel[-1:]*0., traj_vel[:-1]], dim=0)              # shift-1 in dim 0
    traj_vel = torch.cat([traj_vel, traj_vel[:, 0:1]*0.], dim=1)
    traj_vel = quaternion_apply(root_rotZ, traj_vel)[:,0:2]
    transXY = torch.cumsum(traj_vel, dim=0)
    trans = torch.cat([transXY, transZ], dim=-1)                    #(T,3)


    ### poses
    poses_6d = motion[:,10:136].reshape((-1,21,6))
    poses = matrix_to_axis_angle(rotation_6d_to_matrix(poses_6d))   #(T,21,3)
    # print("root_rot, poses, trans", root_rot.shape, poses.shape, trans.shape)   # torch.Size([46, 6]) torch.Size([46, 21, 3]) torch.Size([46, 3])
    poses = torch.cat([root_rot.unsqueeze(1), poses], dim=1).reshape((-1,22*3))       #(T,22,3)

    return trans, poses

def motion_to_bodyparts(motion):
    B,T = motion.shape[:2]

    bp_list = [[3,6,9,12,15], [1,4,7,10], [2,5,8,11], [13,16,18,20], [14,17,19,21]]
    bp_list = [torch.tensor(bp, dtype=int) - 1 for bp in bp_list]
    
    # extract poses_6d, rifke, local_vel    (Note that root traj and foot contact are ignored)
    poses_6d, rifke, local_vel = motion[:,:,10:136], motion[:,:,136:199], motion[:,:,199:265]
    # reshape to have joint axis
    poses_6d, rifke, local_vel = poses_6d.reshape((B,T,21,6)), rifke.reshape((B,T,21,3)), local_vel.reshape((B,T,22,3))#[:,:,1:]
    root_vel2, local_vel = local_vel[:,:,0], local_vel[:,:,1:]

    # spine
    root = torch.cat([motion[:,:,:10], root_vel2, motion[:,:,265:]], dim=-1)
    spine = torch.cat([poses_6d[:,:,bp_list[0]],\
                        rifke[:,:,bp_list[0]],\
                        local_vel[:,:,bp_list[0]]], dim=-1).reshape((B,T,-1))

    leftFoot = torch.cat([poses_6d[:,:,bp_list[1]], \
                        rifke[:,:,bp_list[1]], \
                        local_vel[:,:,bp_list[1]]], dim=-1).reshape((B,T,-1))
    
    rightFoot = torch.cat([poses_6d[:,:,bp_list[2]], \
                        rifke[:,:,bp_list[2]], \
                        local_vel[:,:,bp_list[2]]], dim=-1).reshape((B,T,-1))
    
    leftHand = torch.cat([poses_6d[:,:,bp_list[3]], \
                        rifke[:,:,bp_list[3]], \
                        local_vel[:,:,bp_list[3]]], dim=-1).reshape((B,T,-1))
    
    rightHand = torch.cat([poses_6d[:,:,bp_list[4]], \
                        rifke[:,:,bp_list[4]], \
                        local_vel[:,:,bp_list[4]]], dim=-1).reshape((B,T,-1))
    return root, spine, leftFoot, rightFoot, leftHand, rightHand

def bodyparts_to_smpl(root, spine, leftFoot, rightFoot, leftHand, rightHand):
    B,T = root.shape[:2]
    # process root
    # print(root.shape, spine.shape, leftFoot.shape, rightFoot.shape, leftHand.shape, rightHand.shape)
    ### root traj
    root_rotvelZ, root_rot_Zfiltered = root[:,:,0], root[:,:,1:7]
    traj_vel, transZ = root[:,:,7:9], root[:,:,9:10]
        # process rotation velocity
    root_rotvelZ = torch.cat([root_rotvelZ[:,-1:]*0., root_rotvelZ[:,:-1]], dim=1)  # shift-1 in dim 0
    root_rotZ = torch.cumsum(root_rotvelZ, dim=1)
    root_rotZ = torch.stack([root_rotZ*0., root_rotZ*0., root_rotZ],dim=-1)
    root_rotZ = matrix_to_quaternion(euler_angles_to_matrix(root_rotZ, "XYZ"))

        # reconstruct root_rot
    root_rot_Zfiltered = matrix_to_quaternion(rotation_6d_to_matrix(root_rot_Zfiltered))

    root_rot = quaternion_multiply(root_rotZ, root_rot_Zfiltered)   #(T,4)
    root_rot = matrix_to_axis_angle(quaternion_to_matrix(root_rot))
        # traj
    traj_vel = torch.cat([traj_vel[:,-1:]*0., traj_vel[:,:-1]], dim=1)              # shift-1 in dim 0
    traj_vel = torch.cat([traj_vel, traj_vel[:,:, 0:1]*0.], dim=-1)
    traj_vel = quaternion_apply(root_rotZ, traj_vel)[:,:,0:2]
    transXY = torch.cumsum(traj_vel, dim=1)
    trans = torch.cat([transXY, transZ], dim=-1)                    #(T,3)

    ### process bodyparts
    spine = spine.reshape((B,T,5,12))[:,:,:,:6]
    leftFoot = leftFoot.reshape((B,T,4,12))[:,:,:,:6]
    rightFoot = rightFoot.reshape((B,T,4,12))[:,:,:,:6]
    leftHand = leftHand.reshape((B,T,4,12))[:,:,:,:6]
    rightHand = rightHand.reshape((B,T,4,12))[:,:,:,:6]
    bp_list = [[3,6,9,12,15], [1,4,7,10], [2,5,8,11], [13,16,18,20], [14,17,19,21]]
    rec_bp = torch.zeros((B,T,22,6), device=spine.device)
    rec_bp[:,:,bp_list[0]] = spine
    rec_bp[:,:,bp_list[1]] = leftFoot
    rec_bp[:,:,bp_list[2]] = rightFoot
    rec_bp[:,:,bp_list[3]] = leftHand
    rec_bp[:,:,bp_list[4]] = rightHand
    poses = matrix_to_axis_angle(rotation_6d_to_matrix(rec_bp))
    poses[:,:,0] = root_rot
    return trans, poses


def calc_mean_var(motion):
    motion_mean = motion.mean(dim=0)
    motion_std = motion.std(dim=0)
    motion_std[0:1] = motion_std[0:1].mean() / 1.0 
    motion_std[1:7] = motion_std[1:7].mean() / 1.0
    motion_std[7:9] = motion_std[7:9].mean() / 1.0
    motion_std[9:10] = motion_std[9:10].mean() / 1.0
    motion_std[10:136] = motion_std[10:136].mean() / 1.0
    motion_std[136:199] = motion_std[136:199].mean() / 1.0
    motion_std[199:265] = motion_std[199:265].mean() / 1.0
    motion_std[265:] = motion_std[265:].mean() / 1.0
    return motion_mean, motion_std


class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size=512, beat_nframe=6):
        super().__init__()
        self.batch_size = batch_size

        #### phase model
        self.phase_model = None
        self.beat_nframe = beat_nframe

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

    def setup(self, stage=None):
        with open(f"./processed_data/posetraj_data.pkl", 'rb') as handle:
            motion_dict = pickle.load(handle)

        with open(f"./processed_data/MotionPAE_latent10.pkl", 'rb') as handle:
            codePAE_dict = pickle.load(handle)

        with open(f"./data/hml3d_split_data.pkl", 'rb') as handle:
            split_dict = pickle.load(handle)

        with open(f"./processed_data/text_sentence_all.pkl", 'rb') as handle:
            text_dict = pickle.load(handle)

        rep_all = []
        split_idx_array = []
        textemb_all = []
        for k in motion_dict:
            len_k = motion_dict[k].shape[0]
            if len_k < 40:
                continue
            if len_k > 196:
                len_k = 196

            motion_code_k = codePAE_dict[k] #(1,4,512)
            rep_k = self.latent_reparam(motion_code_k, torch.linspace(-1,1,len_k).unsqueeze(0))[0]
            rep_all.append(rep_k)

            split_idx_array.append(split_dict[k])
            # text_emb in range [1,4], repeat to 12.
            text_emb = text_dict[k]
            repeat_times = int(12 / text_emb.shape[0])
            text_emb = text_emb.unsqueeze(0).expand((repeat_times,-1,-1))
            text_emb = text_emb.reshape((12,512))
            textemb_all.append(text_emb)
        split_idx_array = torch.tensor(split_idx_array).long()

        ### !!! standardize rep
        if Path(f"rep10_mean.npy").exists():
            self.rep_mean = torch.from_numpy(np.load(f"rep10_mean.npy"))
            self.rep_std = torch.from_numpy(np.load(f"rep10_std.npy"))
        else:
            all_rep = torch.cat(rep_all, dim=0)
            self.rep_mean, self.rep_std = all_rep.mean(dim=0), all_rep.std(dim=0)
            np.save(f"rep10_mean.npy", self.rep_mean.detach().cpu().numpy())
            np.save(f"rep10_std.npy", self.rep_std.detach().cpu().numpy())

        # process length
        filter_motion_index = [i for i in range(len(rep_all)) if rep_all[i].shape[0] <= 224 or rep_all[i].shape[0] >= 40]
        rep_all = [rep_all[i] for i in range(len(rep_all)) if i in filter_motion_index]
        textemb_all = [textemb_all[i] for i in range(len(textemb_all)) if i in filter_motion_index]
        split_idx_array = [split_idx_array[i] for i in range(len(split_idx_array)) if i in filter_motion_index]
        split_idx_array = torch.tensor(split_idx_array).float()

        # process into tensordataset format
        lengths_tensor = torch.tensor([pose.shape[0] for pose in rep_all]).long()
        rep_all = pad_sequence(rep_all, batch_first=True).float()
        
        textemb_all = torch.stack(textemb_all, dim=0)

        # normalization
        rep_all = self.transform(rep_all)

        masks_tensor = torch.zeros((rep_all.shape[0], rep_all.shape[1]), device=rep_all.device, dtype=torch.bool)
        progress_tensor = torch.zeros((rep_all.shape[0], rep_all.shape[1]), device=rep_all.device, dtype=torch.float)
        for b in range(len(lengths_tensor)):
            masks_tensor[b,:lengths_tensor[b]] = True
            progress_tensor[b,:lengths_tensor[b]] = torch.linspace(-1,1,lengths_tensor[b])

        
        # for validation
        id_tensor = torch.arange(rep_all.shape[0]).long()

        train_idx = (split_idx_array == 0)
        valid_idx = (split_idx_array == 1)
        test_idx = (split_idx_array == 2)

        print(rep_all.shape, textemb_all.shape)

        # print("max class_index:", torch.amax(class_index_all))    #177
        self.train_dataset = TensorDataset(id_tensor[train_idx], rep_all[train_idx], \
                                           textemb_all[train_idx], masks_tensor[train_idx], progress_tensor[train_idx])
        self.val_dataset = TensorDataset(id_tensor[valid_idx], rep_all[valid_idx], \
                                         textemb_all[valid_idx], masks_tensor[valid_idx], progress_tensor[valid_idx])
        self.test_dataset = TensorDataset(id_tensor[test_idx], rep_all[test_idx], \
                                         textemb_all[test_idx], masks_tensor[test_idx], progress_tensor[test_idx])


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, drop_last=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)