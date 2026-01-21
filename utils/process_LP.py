from pathlib import Path
import torch
import numpy as np
from utils.utils import *
from utils.rotation_conversion import *
from smplx import SMPL
from utils.utils import forward_kinematics, motion_to_smpl

from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from scipy import interpolate, signal

from PIL import Image, ImageDraw
from matplotlib.colors import hsv_to_rgb

import matplotlib
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt

from itertools import product
all_directions = list(product([-1, 0, 1], repeat=3))
all_directions.remove((0,0,0))
all_directions = torch.tensor(all_directions).float()   #(26,3)
all_directions[:,2] *= 3
all_directions = all_directions / all_directions.norm(p=2, dim=-1, keepdim=True)
# all_directions = all_directions.cuda()

########
#   process motion to contLP
#######
def get_reference_vector(all_joint_pos):
    right_vec = all_joint_pos[:, 1] - all_joint_pos[:, 2]  # "left_hip" - "right_hip"
    up_vec = torch.tensor([0, 0, 1], device=right_vec.device, dtype=right_vec.dtype).unsqueeze(0)  # (1,3)
    up_vec = up_vec.expand(right_vec.shape)

    forward_vec = torch.cross(right_vec, up_vec)  # (T,3)
    all_refvec = torch.stack([right_vec, forward_vec, up_vec], dim=1)
    all_refvec = all_refvec / torch.norm(all_refvec, dim=2, keepdim=True)
    return all_refvec

def prpp(all_joint_pos, all_refvec, joint_id, parent_id):
    ### spatial
    prpp = all_joint_pos[:, joint_id] - all_joint_pos[:, parent_id] #(T,3)
    prpp = torch.sum(prpp.unsqueeze(1) * all_refvec, dim=-1)        #(T,1,3) * (T,3,3) -> (T,3,3) -> (T,3)
    # prpp = prpp[:-1]
    return prpp

def feature_extraction(all_joint_pos):
    # get reference vector
    all_refvec = get_reference_vector(all_joint_pos)  # (T,3,3)
    RT_spat = all_refvec[:,1]   # forward vec
    ####
    # LBN extraction
    ####
    #   bodypart orientation
    LH_spat = prpp(all_joint_pos, all_refvec, 20, 16)
    RH_spat = prpp(all_joint_pos, all_refvec, 21, 17)
    LL_spat = prpp(all_joint_pos, all_refvec, 10, 1)
    RL_spat = prpp(all_joint_pos, all_refvec, 11, 2)
    SP_spat = prpp(all_joint_pos, all_refvec, 12, 0)
    
    spat_val = torch.stack([RT_spat, LL_spat, RL_spat, SP_spat, LH_spat, RH_spat], dim=1)
    return spat_val

def motion_to_contLP(rec_motion):
    rec_traj, rec_pose = motion_to_smpl(rec_motion)
    return posetraj_to_contLP(rec_pose)

def posetraj_to_contLP(rec_pose):
    rec_joint_pos = forward_kinematics(torch.zeros_like(rec_pose[:,:3]), rec_pose.reshape((-1, 22, 3)))
    rec_spat_val = feature_extraction(rec_joint_pos)
    return rec_spat_val

########
#   process contLP to discLP
#######

def spatial_LP_discretization(cont_LP, soft=False):
    retval = cont_LP / torch.norm(cont_LP, p=2, dim=-1, keepdim=True)  # unit length

    if not soft:
        preds = retval @ all_directions.t()            #(T,5,26)
        cont_LP_weight = torch.argmax(preds, dim=-1)    #(T,5)
        cont_LP_weight = torch.nn.functional.one_hot(cont_LP_weight, num_classes=all_directions.shape[0])
        retval = cont_LP_weight.float() @ all_directions        #(T,5,3)
        retval = retval / torch.norm(retval, p=2, dim=-1, keepdim=True)

    return retval

def temporal_saliency_extraction(cont_LP):
    cont_LP_normed = cont_LP / torch.norm(cont_LP, p=2, dim=-1, keepdim=True)  # unit length
    LP_fn = cont_LP_normed @ all_directions.t()            #(T,5,26)
    LP_fn = torch.gradient(LP_fn, dim=0)[0]
    
    # b, a = signal.butter(3, 2, fs=30)
    # preds = torch.tensor(
    #     signal.filtfilt(b, a, preds.detach().cpu().numpy(), axis=0).copy()
    # ).float()
    # LP_fn = torch.amax(preds, dim=-1)    #(T,5)
    
    LP_weight_all = []
    for p in range(LP_fn.shape[1]):
        fn = LP_fn[:,p]
        LP_weight = global_frame_importance_extraction(fn.detach().cpu().numpy())
        LP_weight_all.append(torch.from_numpy(LP_weight))
    return torch.stack(LP_weight_all, dim=1)

def generate_LP_mask(cont_LP, num_tokens_per_bodypart=None, saliency_threshold=None):
    LP_weight = temporal_saliency_extraction(cont_LP)   #(T,6)
    LP_weight = LP_weight.detach().cpu().numpy()
    # obtain discretized LP for each bodypart
    LP_mask = []
    for bp in range(LP_weight.shape[1]):
        # process LP_weight to select keyframes
        weight_bp = LP_weight[:,bp]
        if num_tokens_per_bodypart is not None:
            assert type(num_tokens_per_bodypart) == int
            order = np.argsort(-weight_bp)  # sort descending order
            idx_order = np.arange(weight_bp.shape[0])[order]
            idx_order = idx_order[:num_tokens_per_bodypart]

            bool_mask = np.zeros((weight_bp.shape[0], 1))
            bool_mask[idx_order] += 1
        else:
            assert type(saliency_threshold) == float
            bool_mask = weight_bp > saliency_threshold
            bool_mask = bool_mask.float()
            
        LP_mask.append(bool_mask)
    LP_mask = np.stack(LP_mask, axis=1)
    return torch.from_numpy(LP_mask)
    
####################
# agglomerative clustering helpers
####################

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

def agglomerative_clustering_time_series(temp_fn, debug=False):
    ## agglo
    grid = grid_to_graph(n_x=temp_fn.shape[0], n_y=1, n_z=1)
    # Instantiate the clustering object         # linkage='complete' / 'ward'
    model = AgglomerativeClustering(
        n_clusters=None, connectivity=grid, distance_threshold=0, compute_full_tree=True, compute_distances=True, linkage='ward'
    )

    if len(temp_fn.shape) == 1:
        temp_fn = temp_fn[:, np.newaxis]
    model.fit(temp_fn)

    # print(temp_fn.shape)    #(84,5)
    # print("children")
    # print(model.children_)  #(83,2)
    # print("distances")
    # print(model.distances_) #(83,)
    
    ### interpret dendrogram into segment boundaries
    weight = np.zeros(temp_fn.shape[0])
    segment_keyframe = list(range(temp_fn.shape[0]))
    for i, node in enumerate(model.children_):
        keyframes = [segment_keyframe[idx] for idx in node]     # get valid frame idx 
        # smaller idx is selected keyframe, larger one is trimmed and process into weight
        ordered_keyframe = [keyframes[0],keyframes[1]] if keyframes[0] < keyframes[1] else [keyframes[1],keyframes[0]]
        segment_keyframe.append(ordered_keyframe[0])
        # weight[ordered_keyframe[1]] = model.distances_[i]

        # regularize distance to be increasing (by record child weight)
        max_child_weight = max(weight[ordered_keyframe[0]]+1e-5, weight[ordered_keyframe[1]]+1e-5, model.distances_[i])
        weight[ordered_keyframe[1]] = max_child_weight
        weight[ordered_keyframe[0]] = max_child_weight
        model.distances_[i] = max_child_weight  # for visualization

    weight[0] = 0.
    # reassign the weight of start frame (0) and end frame (-1)
    # w_max, w_min = weight[1:-1].max(), weight[1:-1].min()
    # weight[0] = w_max + w_min
    # weight[-1] = w_max + w_min * 2 

    # temperature = 1
    # weight = scipy.special.softmax(weight * temperature)

    # order = np.argsort(weight)
    # ordered_weight = weight[order]
    # idx_order = np.arange(temp_fn.shape[0])[order]
    # print(list(zip(idx_order, ordered_weight)))
    # plt.plot(weight)
    # plt.show()
    
    # plot_dendrogram(model)
    # plt.show()

    return weight

def global_frame_importance_extraction(temp_fn):
    ### count movement activation of lbn columns
    # temp_fn_mean = np.mean(temp_fn, axis=0, keepdims=True)   #(1,J)
    # temp_fn_std = np.std(temp_fn, axis=0, keepdims=True)     #(1,J)
    # temp_fn_norm = (temp_fn - temp_fn_mean) / temp_fn_std
    
    boundaries_all = agglomerative_clustering_time_series(temp_fn, debug=False)
    return boundaries_all
    

####################
# visualization
####################

def signal_to_color(h, s=1, v=1):
    h = h % 1.0
    # v = prpp[:,2] + 1.
    # v = np.ones_like(h) * v
    # s = np.ones_like(h) * s
    hsv = np.stack([h, s, v], axis=-1)
    rgb = hsv_to_rgb(hsv)
    return rgb

def spatial_color_for_visualization(concat_signals):
    colored_values = []

    for i in range(concat_signals.shape[1]):
        prpp = concat_signals[:, i, :3]
        h = 0.5 + np.arctan2(prpp[:, 0], prpp[:, 1]) / np.pi / 2  
        # s = 1 - np.exp(-np.linalg.norm(prpp[:, :2], axis=1) / 0.042)  #sqrt(2*(0.03)^2) ~ 0.042
        # h = h + 0.5 if i >= 4 else h
        
        control = (prpp[:, 2] > 0).float()
        v = control * 1 + (1-control) * (prpp[:, 2]/2 + 1)    # 1 if prpp[:, 2] > 0 else prpp[:, 2] + 1   
        s = (1-control) * 1 + control * (1 - prpp[:, 2]/2)    # 1 if prpp[:, 2] <= 0 else 1 - prpp[:, 2]

        colored_value = signal_to_color(h, s=s, v=v)
        colored_values.append(colored_value)
    
    return np.stack(colored_values, axis=1)     #(T, E, 3)


# axis order: [right, forward, up]
global left_symbols, right_symbols
left_symbols = None
right_symbols = None

def load_symbol_png():
    global left_symbols, right_symbols
    if left_symbols is not None:
        return left_symbols, right_symbols

    left_symbols = [[[None for _ in range(3)] for _ in range(3)] for _ in range(3)]
    right_symbols = [[[None for _ in range(3)] for _ in range(3)] for _ in range(3)]

    root_folder_path = Path("./visualization/symbol_icon/basic")
    filenames_to_idx = [['Laban-diagonal-back-high-5', None, [0,-1,1]],
                        ['Laban-diagonal-back-high-6', [0,-1,1], None],
                        ['Laban-diagonal-back-high-left', [-1,-1,1], [-1,-1,1]],
                        ['Laban-diagonal-back-high-right', [1,-1,1], [1,-1,1]],
                        ['Laban-diagonal-back-low-5', None, [0,-1,-1]],
                        ['Laban-diagonal-back-low-6', [0,-1,-1], None],
                        ['Laban-diagonal-back-low-left', [-1,-1,-1], [-1,-1,-1]],
                        ['Laban-diagonal-back-low-right', [1,-1,-1], [1,-1,-1]],
                        ['Laban-diagonal-back-middle-5', None, [0,-1,0]],
                        ['Laban-diagonal-back-middle-6', [0,-1,0], None],
                        ['Laban-diagonal-back-middle-left', [-1,-1,0], [-1,-1,0]],
                        ['Laban-diagonal-back-middle-right', [1,-1,0], [1,-1,0]],
                        #
                        ['Laban-diagonal-forward-high-1', None, [0,1,1]],
                        ['Laban-diagonal-forward-high-10', [0,1,1], None],
                        ['Laban-diagonal-forward-high-left', [-1,1,1], [-1,1,1]],
                        ['Laban-diagonal-forward-high-right', [1,1,1], [1,1,1]],
                        ['Laban-diagonal-forward-low-1', None, [0,1,-1]],
                        ['Laban-diagonal-forward-low-10', [0,1,-1], None],
                        ['Laban-diagonal-forward-low-left', [-1,1,-1], [-1,1,-1]],
                        ['Laban-diagonal-forward-low-right', [1,1,-1], [1,1,-1]],
                        ['Laban-diagonal-forward-middle-1', None, [0,1,0]],
                        ['Laban-diagonal-forward-middle-10', [0,1,0], None],
                        ['Laban-diagonal-forward-middle-left', [-1,1,0], [-1,1,0]],
                        ['Laban-diagonal-forward-middle-right', [1,1,0], [1,1,0]],
                        #
                        ['Laban-diagonal-high-left', [-1,0,1], [-1,0,1]],
                        ['Laban-diagonal-high-place', [0,0,1], [0,0,1]],
                        ['Laban-diagonal-high-right', [1,0,1], [1,0,1]],
                        ['Laban-diagonal-low-left', [-1,0,-1], [-1,0,-1]],
                        ['Laban-diagonal-low-place', [0,0,-1], [0,0,-1]],
                        ['Laban-diagonal-low-right', [1,0,-1], [1,0,-1]],
                        ['Laban-diagonal-middle-left', [-1,0,0], [-1,0,0]],
                        ['Laban-diagonal-middle-place', [0,0,0], [0,0,0]],
                        ['Laban-diagonal-middle-right', [1,0,0], [1,0,0]],
                        ]

    for img, left_idx, right_idx in filenames_to_idx:
        background = Image.new(mode="RGBA", size=(18,40), color=(255, 255, 255))
        symbol = Image.open(root_folder_path/f"{img}.png")    # was (1920x1080)
        symbol.thumbnail((18,40), Image.ANTIALIAS)
        symbol = symbol.convert("RGBA")
        # mask = Image.new(mode="1", size=(18,40), color=1)
        # print(np.array(symbol).shape)#, np.array(mask).shape)
        # mask = Image.fromarray(np.array(symbol)[:,:,2])
        # print(np.array(mask).shape)#, np.array(mask).shape)
        background.paste(symbol, (0,0))
        # background.show()
        # symbol.convert("RGB")
        # print()
        # symbol.show()

        if left_idx is not None:
            a,b,c = left_idx
            left_symbols[a][b][c] = background
        
        if right_idx is not None:
            a,b,c = right_idx
            right_symbols[a][b][c] = background

    return left_symbols, right_symbols
    

def LP_visualization(hint_b, hint_mask_b):
    left_symbols, right_symbols = load_symbol_png()
    # print(hint_b.shape, hint_mask_b.shape)  # (T,6,3), (T,6)
    # each symbol in (40, 60), 150 height per 30 frames
    time = hint_b.shape[0]
    # print(240, int(60 + time / 30 * 150), time)
    height = int(60 + time / 30 * 300)
    background = Image.new(mode="RGBA", size=(120, int(60 + time / 30 * 300)), color=(255, 255, 255))

    all_dir = list(product([-1, 0, 1], repeat=3))
    all_dir.remove((0,0,0))
    all_dir = torch.tensor(all_dir).float()   #(26,3)

    # joint order was [RT, LL, RL, SP, LH, RH]
    for idx, joint in enumerate([0,4,1,2,5,3]):
        hint_bj = hint_b[:,joint]                #(T,3)
        hint_mask_bj = hint_mask_b[:,joint]     #(T,)
        if hint_mask_bj.sum() == 0:
            continue
        bp_keyframes = np.nonzero(hint_mask_bj)[:,0]
        # find associated symbol
        for t in bp_keyframes:
            hint_btj = hint_bj[t]   #(3,)
            # print(hint_btj, np.sign(hint_btj))
            a,b,c = [int(i) for i in np.sign(hint_btj)]
            if joint in [0,4,1]:
                # left
                symbol = left_symbols[a][b][c]
            else:
                # right
                symbol = right_symbols[a][b][c]

            background.paste(symbol, (idx * 20, height - int(t / 30 * 300)))

    
    background1 = ImageDraw.Draw(background)
    # horizontal line 
    for l in range(30, height, 300):
        background1.line((40,height-l,80,height-l), fill=(0,0,0,255))
    # vertical line
    background1.line((40,height-30,40,0), fill=(0,0,0,255))
    background1.line((80,height-30,80,0), fill=(0,0,0,255))
    return background
    # background.show()
    # background.save(root_path / f'{folder}_concat.png', 'PNG')