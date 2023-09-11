#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==========================================================
# Created by : Mohit Anand
# Created on : Fri Mar 17 2023 at 10:43:09 PM
# ==========================================================
# Created on Fri Mar 17 2023
# __copyright__ = Copyright (c) 2023, Mohit Anand's Project
# __credits__ = [Mohit Anand,]
# __license__ = Private
# __version__ = 0.0.0
# __maintainer__ = Mohit Anand
# __email__ = itsmohitanand@gmail.com
# __status__ = Development
# ==========================================================

import numpy as np
from torch.utils.data.dataset import Dataset
import torch
from fomo.io import read_benchmark_data
from typing import Tuple


if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

class MultiModalDatasets(Dataset):
    def __init__(self, ds_path, pft="beech", split_type="train", xs_list = ["age", "sv", "laicum", "h", "d"], xs_year_before=3, xd_year_from=2, target = 'MBR', agg="monthly", classify=False ) -> None:
        super().__init__()

        self.hist_norm = 3308488.0
        forest_data = read_benchmark_data(ds_path, pft = pft, xs_list=xs_list, xs_year_before=xs_year_before, xd_year_from=xd_year_from, target = target, agg=agg, classify = classify  )

        self.xd_m = torch.tensor(forest_data[split_type][0].astype(np.float32)).cpu().detach()
        self.xs_m = torch.tensor((forest_data[split_type][1][:,:,:,np.newaxis]/self.hist_norm).astype(np.float32)).cpu().detach() # ! This number is computed from all the input and is almost equal to maxinum number of trees ever

        self.labels = torch.tensor(forest_data[split_type][2][:].astype(np.float32)).cpu().detach()
        
        self.n_modalities = self.xd_m.shape[2] + self.xs_m.shape[2]

        self.bins = forest_data["bins"]
        self.mean_Xd = forest_data["mean_Xd"]
        self.std_Xd = forest_data["std_Xd"]

    def get_n_modalities(self)->int:
        return self.n_modalities
    
    def get_seq_len(self)->Tuple:
        return tuple((self.xd_m.shape[1] for i in range(self.xd_m.shape[2]))) + tuple((self.xs_m.shape[1] for i in range(self.xs_m.shape[2])))

    def get_dim(self)-> Tuple:
        return tuple((1 for i in range(self.xd_m.shape[2]))) +  tuple((1 for i in range(self.xs_m.shape[2])))
    
    def get_lbl_info(self)-> Tuple:
        return self.labels.shape[0], self.labels.shape[1]
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        X = tuple((self.xd_m[index, :, i] for i in  range(self.xd_m.shape[2]))) + tuple((self.xs_m[index, :, i] for i in  range(self.xs_m.shape[2])))
        Y = self.labels[index]
        return X, Y
    