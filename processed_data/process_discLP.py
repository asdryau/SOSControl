import torch
import pickle
import numpy as np

from utils.utils import *

import matplotlib
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt

from utils.process_LP import spatial_LP_discretization, temporal_saliency_extraction

if __name__ == "__main__":

    with open("./processed_data/contLP_data.pkl", "rb") as file: 
        contLP_all = pickle.load(file)

    # process posetraj format
    discLP_all = {}
    LP_weight_all = {}
    for k in contLP_all:
        contLP = contLP_all[k]
        discLP = spatial_LP_discretization(contLP, soft=False)
        LP_weight = temporal_saliency_extraction(contLP).float()
        print(k, contLP.shape, discLP.shape, LP_weight.shape)
        discLP_all[k] = discLP.detach()
        LP_weight_all[k] = LP_weight

    with open(f"./processed_data/discLP_full_data.pkl", 'wb') as handle:
        pickle.dump(discLP_all, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f"./processed_data/discLP_weight_data.pkl", 'wb') as handle:
        pickle.dump(LP_weight_all, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # #####################
    # with open("./processed_data/contLP_data.pkl", "rb") as file: 
    #     contLP_all = pickle.load(file)

    # # normalize
    # all_contLP = []
    # for k in contLP_all:
    #     contLP = contLP_all[k]
    #     all_contLP.append(contLP.float())

    # all_contLP = torch.cat(all_contLP, dim=0)
    # contLP_mean, contLP_std = all_contLP.mean(dim=0), all_contLP.std(dim=0)
    # np.save(f"contLP_mean.npy", contLP_mean.detach().cpu().numpy())
    # np.save(f"contLP_std.npy", contLP_std.detach().cpu().numpy())

    # #####################
    # with open("./processed_data/contLP_data.pkl", "rb") as file: 
    #     contLP_all = pickle.load(file)

    # contLP_mean = torch.from_numpy(np.load(f"contLP_mean.npy"))
    # contLP_std = torch.from_numpy(np.load(f"contLP_std.npy"))

    # # process posetraj format
    # LP_weight_all = {}
    # for k in contLP_all:
    #     contLP = contLP_all[k]
    #     contLP = (contLP - contLP_mean.to(contLP.device)) / (contLP_std.to(contLP.device) + 1e-8)
    #     LP_weight = temporal_saliency_extraction(contLP).float()
    #     print(k, contLP.shape, LP_weight.shape)
    #     LP_weight_all[k] = LP_weight

    # with open(f"./processed_data/discLP_normweight_data.pkl", 'wb') as handle:
    #     pickle.dump(LP_weight_all, handle, protocol=pickle.HIGHEST_PROTOCOL)

    





