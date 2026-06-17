import os.path
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from util.timefeatures import time_features
from data.mapEQ_query import StationWeightLookup

class NPZ_PCA_Fenlei_Dataset_train(Dataset):
    def __init__(self, pkl_path='./dataset/down_pca_train.pkl'):
        with open(pkl_path, 'rb') as f:
            self.filelist = pickle.load(f)
    def __len__(self):
        return len(self.filelist)
    def get_id(self, data):
        if "sample_id" in data:
            val = data["sample_id"]
        elif "id" in data:
            val = data["id"]
        else:
            raise KeyError("No 'sample_id' or 'id' found in data.")
        return val.item() if isinstance(val, np.ndarray) else val
    def __getitem__(self, idx):
        filepath = self.filelist[idx]
        data = np.load(filepath, allow_pickle=True)
        seq_x = data["seq_x"]      # (1008, 3)
        seq_x_mark = data["seq_x_mark"]
        label = data["label"].item() if isinstance(data["label"], np.ndarray) else data["label"]
        idset = self.get_id(data)
        date = data["date"].item() if isinstance(data["date"], np.ndarray) else data["date"]
        p0 = self.weight_lookup.get_p0(idset)
        return seq_x, seq_x_mark, label, idset, date,p0

class NPZ_PCA_Fenlei_Dataset_test(Dataset):
    def __init__(self, pkl_path='./dataset/down_pca_test.pkl'):
        self.weight_lookup=StationWeightLookup(station_info_csv='/openbayes/home/dataset/StationInfo.csv', weight_grid_pkl=weight_lookup)
        with open(pkl_path, 'rb') as f:
            self.filelist = pickle.load(f)
    def __len__(self):
        return len(self.filelist)
    def __getitem__(self, idx):
        filepath = self.filelist[idx]
        data = np.load(filepath, allow_pickle=True)
        seq_x = data["seq_x"]      # (1008, 3)
        seq_x_mark = data["seq_x_mark"]
        label = data["label"].item() if isinstance(data["label"], np.ndarray) else data["label"]
        idset = data["id"].item() if isinstance(data["id"], np.ndarray) else data["id"]
        date = data["date"].item() if isinstance(data["date"], np.ndarray) else data["date"]
        p0 = self.weight_lookup.get_p0(idset)
        return seq_x, seq_x_mark, label, idset, date,p0