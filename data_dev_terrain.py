# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 16:44:49 2021

Development done of terrain data files - amont_dopage2 and aval_dopage2 at SAUR

@author: Admin
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dataread_terrain as d_terr

root = r'D:\VP\Viewpoint_data\DATA_terrain'
os.chdir(root)
files = [f for f in os.listdir() if '.csv' in f]


colors = [
    '#42f5e0','#12aefc','#1612fc','#6a00a3',
    '#8ef743','#3c8f01','#0a4001','#fc03ca',
    '#d9d200','#d96c00','#942c00','#fc2803',
    '#e089b6','#a3a3a3','#7a7a7a','#303030'
    ]


specie = {'E': 'Erpobdella','G':'Gammarus','R':'Radix'}
species = 'R'

dfs,dfs_mean = d_terr.read_data_terrain(files)
df = dfs[species]

dope_start = pd.Timestamp(year = 2021,month = 1, day = 14, hour = 12)
dope_end = pd.Timestamp(year = 2021,month = 1, day = 14, hour = 15)
date_range = [dope_start,dope_end]

# plot
fig,axe = d_terr.single_plot(df, species)


# this does not work
timestep = 60*60    # 10 minute
start_time = df.index[0]
df['timestep'] = ((df.index - start_time).total_seconds() // timestep).astype(int)

df_mean = df.groupby(['timestep']).mean()
index_map = dict(zip(df['timestep'].unique(),[df[df['timestep'] == i].index[-1] for i in df['timestep'].unique()]))
df_mean = df_mean.set_index(df_mean.index.map(index_map))

fig,axe = d_terr.single_plot(df_mean, species, 'Groupby means')

fig,axe = d_terr.plot_16(df_mean)

"""
Agorithm that checks the last two hours average of movement
"""

#test mahalanobis distance
    
## %% functions
def cov_matrix(data):
    covariance_matrix = np.cov(data, rowvar=False)
    inv_covariance_matrix = np.linalg.inv(covariance_matrix)
    return covariance_matrix,inv_covariance_matrix

def MahalanobisDist(inv_cov,mean,data):
    diff = data - mean
    md = []
    for i in range(len(diff)):
        md.append(np.sqrt(diff[i].dot(inv_cov).dot(diff[i])))
    return md

def EuclideanDist(mean,data):
    diff = data - mean
    ed = []
    for i in range(len(diff)):
        ed.append(np.linalg.norm(diff[i]))
    return ed
    

"""
If this file is unarchived, dead should be removed properly
"""
    
#Take 24 hours of normal behaviour - 13 jan 
df_train = df_mean[df_mean.index.day == 13]
means = df_train.mean()
df_mean = df_mean[[1,3,5,6,7,10,11,12,13,15,16]]

#reform train with correct index
df_mean = df_mean[df_mean.index.day >= 13]
df_train = df_mean[df_mean.index.day == 13]

data = np.array(df_mean.values)
data_train = np.array(df_train.values)

#calculate covariance matrices
cov,inv_cov = cov_matrix(data_train)
mean_distr = data_train.mean(axis = 0)

#Euclidean distance
df_mean['Eucl'] = EuclideanDist(mean_distr, data)
fig,axe = d_terr.combined_plot(df_mean.Eucl,species,'L2 distance')
axe.axvspan(date_range[0], date_range[1], alpha=0.7, color='orange')

#Mahalanobis distance
df_mean['MD'] = MahalanobisDist(inv_cov, mean_distr, data)
fig,axe = d_terr.combined_plot(df_mean.MD, species, 'Mahalanobis dist')
axe.axvspan(date_range[0], date_range[1], alpha=0.7, color='orange')


#%% with standardised values

#standardise dataframe
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
data_x = min_max_scaler.fit_transform(data)

df_mean_x = pd.DataFrame(data_x,index = df_mean.index, columns = df_mean.columns[:-2])
df_train_x = df_mean_x[df_mean_x.index.day == 13]

data_train_x = np.array(df_train_x.values)

cov_x,inv_cov_x = cov_matrix(data_train_x)
mean_distr_x = data_train_x.mean(axis = 0)

#Euclidean distance scaler (x)
df_mean_x['Eucl'] = EuclideanDist(mean_distr_x, data_x)
fig,axe = d_terr.combined_plot(df_mean_x.Eucl,species,'L2 distance standardised')
axe.axvspan(date_range[0], date_range[1], alpha=0.7, color='orange')

#Mahalanobis distance scaled (x)
df_mean_x['MD'] = MahalanobisDist(inv_cov_x, mean_distr_x, data_x)
fig,axe = d_terr.combined_plot(df_mean_x.MD, species, 'Mahalanobis dist standardised')
axe.axvspan(date_range[0], date_range[1], alpha=0.7, color='orange')