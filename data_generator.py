# -*- coding: utf-8 -*-
# Benchmark data generator for example
# @Time    : 2024/1/1
# @Author  : Juyeon Park (wndus1712@gmail.com)

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import mat73
import scipy

def benchmark_generator(df_name):
    # Load Dataset
    if df_name in ["http", "smtp"]:
        mat = mat73.loadmat('./data/{}.mat'.format(df_name))
    elif df_name in ['heart']:
        # Title of Database: SPECTF heart data 
        train_ = pd.read_csv(f'./data/SPECTF.train', header=None)  # Class Distribution; '0':'1' = 40:40
        test_ = pd.read_csv(f'./data/SPECTF.test', header=None)    # Class Distribution; '0':'1' = 15:172
        dataset = pd.concat((train_, test_), ignore_index=True)    # Use train and test
        mat = {"X":np.array(dataset.iloc[:, 1:]),
               "y":np.array(dataset.iloc[:, 0].replace({1:0, 0:1}))}
    else:
        mat = scipy.io.loadmat('./data/{}.mat'.format(df_name))
    scaler = StandardScaler()
    X = scaler.fit_transform(mat["X"])
    y = mat["y"].astype(int).reshape(-1)
    
    return X, y
