import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import pickle

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

class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size=512, beat_nframe=6):
        super().__init__()
        self.batch_size = batch_size

        #### phase model
        self.phase_model = None
        self.beat_nframe = beat_nframe

    def setup(self, stage=None):

        with open(f"./processed_data/posetraj_data.pkl", 'rb') as handle:
            motion_dict = pickle.load(handle)

        with open(f"./data/hml3d_split_data.pkl", 'rb') as handle:
            split_dict = pickle.load(handle)

        motion_all = []
        split_idx_array = []
        for k in motion_dict:
            if motion_dict[k].shape[0] < 40:
                continue
            motion = motion_dict[k]
            motion = motion[:196]
            motion_all.append(motion)
            split_idx_array.append(split_dict[k])
        split_idx_array = torch.tensor(split_idx_array).long()

        lengths_tensor = torch.tensor([pose.shape[0] for pose in motion_all]).long()
        motion_all = pad_sequence(motion_all, batch_first=True).float()
        masks_tensor = torch.zeros((motion_all.shape[0], motion_all.shape[1]), device=motion_all.device, dtype=torch.bool)
        progress_tensor = torch.zeros((motion_all.shape[0], motion_all.shape[1]), device=motion_all.device, dtype=torch.float)
        for b in range(len(lengths_tensor)):
            masks_tensor[b,:lengths_tensor[b]] = True
            progress_tensor[b,:lengths_tensor[b]] = torch.linspace(-1,1,lengths_tensor[b])
        
        # for validation
        id_tensor = torch.arange(motion_all.shape[0]).long()

        train_idx = (split_idx_array == 0)
        valid_idx = (split_idx_array == 1)

        print(motion_all.shape)

        # print("max class_index:", torch.amax(class_index_all))    #177
        self.train_dataset = TensorDataset(id_tensor[train_idx], motion_all[train_idx], masks_tensor[train_idx], progress_tensor[train_idx])
        self.val_dataset = TensorDataset(id_tensor[valid_idx], motion_all[valid_idx], masks_tensor[valid_idx], progress_tensor[valid_idx])


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, drop_last=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)