import torch
import pickle

from utils.utils import *

import matplotlib
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt

if __name__ == "__main__":

    with open("./processed_data/posetraj_data.pkl", "rb") as file: 
        motion_dict = pickle.load(file)

    # process posetraj format
    k_all = []
    for k in motion_dict:
        k_all.append(k)

    with open(f"./processed_data/k_data.pkl", 'wb') as handle:
        pickle.dump(k_all, handle, protocol=pickle.HIGHEST_PROTOCOL)