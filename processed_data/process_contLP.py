import torch
import pickle

from utils.process_LP import motion_to_contLP

import matplotlib
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt

if __name__ == "__main__":

    with open("./processed_data/posetraj_data.pkl", "rb") as file: 
        motion_dict = pickle.load(file)

    # process posetraj format
    contLP_all = {}
    for k in motion_dict:
        motions = motion_dict[k]
        contLP = motion_to_contLP(motions)
        print(k, motions.shape, contLP.shape)
        contLP_all[k] = contLP.detach()

    with open(f"./processed_data/contLP_data.pkl", 'wb') as handle:
        pickle.dump(contLP_all, handle, protocol=pickle.HIGHEST_PROTOCOL)