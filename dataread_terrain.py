# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 16:44:49 2021

Read terrain data

@author: Admin
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
from data_merge import merge_dfs

#from perftimer import Timer
#t = Timer()


"""
Directory to only include files from one experiment.
"""

def read_data_TOX(dir):
    """
    Parameters
    ----------
    dir : directory
        Contains csv files output from ToxMate (NOT VPCore2).
        Probably one aval, amont

    Returns
    -------
    dataframes - coherent to the form of the dataframe in read_data_VPCore2.
    """
    return None

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

print('The following files will be merged:')
print(files)

dfs = []

for file in files:
    df = pd.read_csv(file,sep = '\t')
    
    print('Erpo: {}'.format(len(df[df['specie'] == 'Erpobdella'])))
    print('Radix: {}'.format(len(df[df['specie'] == 'Radix'])))
    print('Gammare: {}'.format(len(df[df['specie'] == 'Gammarus'])))
    
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values(by = ['time','replica'])
    
    df = df.reset_index(drop = True)
    
    print('Before adjustment: total rows{}'.format(len(df)))
    
    end_time = df.iloc[-1]['time']
    end_data = df[df['time'] == end_time]
    if len(end_data) % 16:
        df = df.iloc[:-len(end_data)]
    
    start_time = df.iloc[0]['time']
    start_data = df[df['time'] == start_time]
    if len(start_data) % 16:
        df = df.iloc[len(start_data):]
        
    start_index = 0
    time_intervals = df.iloc[:48]['time'].unique()
    if len(time_intervals) > 1:
        diffs = []
        for i in range(3):
            diffs.append(df.iloc[16*(i+1)]['time'].second - df.iloc[16*(i)]['time'].second)
            
        diff_dic = dict(zip(diffs,[0,1,2]))
        start_index = 16*diff_dic[max(diffs)]
        
    if start_index:
        df = df.iloc[start_index:]
        
    print('After adjustment: total rows{}'.format(len(df)))
    
    dfs.append(df)
  
# no need to merge - they are in seperate locations!    
df = dfs[0]

#instead of max rows, split into three datasets
df_gamm = df[df['specie'] == 'Gammarus']
df_erpo = df[df['specie'] == 'Erpobdella']
df_radi = df[df['specie'] == 'Radix']


# #create animal column from location E01
df['animal'] = df['replica']

# df['dose'] = '72ug/L'
# df['etude'] = 'ETUDE001'
# df['lot'] = 'Zn'


# #species is input
df = df[df['specie'] == specie[species]]

#total distance inadist is only zeros?
df['dist'] = df['inadist'] + df['smldist'] + df['lardist']
if species == 'R':
    df['dist'] = np.where((df['dist'] > 50),0,df['dist'])

#doping
dope_start = pd.Timestamp(year = 2021,month = 1, day = 14, hour = 12)
dope_end = pd.Timestamp(year = 2021,month = 1, day = 14, hour = 15)
date_range = [dope_start,dope_end]

#plot all animals
fig = plt.figure()
axe = fig.add_axes([0.1,0.1,0.8,0.8])
for i in range(16):
    axe.plot(df[df['animal']==(i+1)]['time'],df[df['animal']==(i+1)]['dist'],label = '{}{}'.format(species,i+1),color = colors[i])
axe.set_title('Mean values')
plt.legend()

timestamps = df['time'].unique()
animals = list(range(1,17))   
df_dist = pd.DataFrame(index = timestamps)

# values assumes that there is the perfect amount
# there should be a way of matching values (concatenate)

for i in animals:
    temp_df = df[df['animal'] == i][['time','dist']]
    temp_df = temp_df.set_index('time')
    temp_df.index.name = None
    temp_df.columns = [i]
    df_dist = df_dist.join(temp_df)
    
temp = df_dist.head(500)


# this does not work
timestep = 60*60    # 10 minute
start_time = df_dist.index[0]
df_dist['timestep'] = ((df_dist.index - start_time).total_seconds() // timestep).astype(int)

df_dist_mean = df_dist.groupby(['timestep']).mean()
index_map = dict(zip(df_dist['timestep'].unique(),[df_dist[df_dist['timestep'] == i].index[-1] for i in df_dist['timestep'].unique()]))
df_dist_mean = df_dist_mean.set_index(df_dist_mean.index.map(index_map))

fig_mean = plt.figure()
axe_mean = fig_mean.add_axes([0.1,0.1,0.8,0.8])
for i in animals:
    axe_mean.plot(df_dist_mean.index,df_dist_mean[i],color = colors[i-1],label = '{}{}'.format(species,i))
plt.legend()
    

fig,axe = plt.subplots(4,4,sharex = True, figsize = (20,12))
for i in animals:
    axe[(i-1)//4,(i-1)%4].plot(df_dist_mean.index,df_dist_mean[i],color = colors[i-1])
    
# extract dead values - in the future this could be done with random samples
# or a check every 2 hours

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
    
    
#Take 24 hours of normal behaviour - 13 jan 

df_train = df_dist_mean[df_dist_mean.index.day == 13]
means = df_train.mean()
#df_dist_mean = df_dist_mean[means[means > 5].index]
df_dist_mean = df_dist_mean[[1,3,5,6,7,10,11,12,13,15,16]]

#remove earlier dates
df_dist_mean = df_dist_mean[df_dist_mean.index.day >= 13]
df_train = df_dist_mean[df_dist_mean.index.day == 13]

data = np.array(df_dist_mean.values)
data_train = np.array(df_train.values)


cov,inv_cov = cov_matrix(data_train)
mean_distr = data_train.mean(axis = 0)

#Euclidean distance
df_dist_mean['Eucl'] = EuclideanDist(mean_distr, data)

fig = plt.figure(figsize = (15,8))
axe = fig.add_axes([0.1,0.1,0.8,0.8])
axe.set_title('L2 normal distance')
axe.plot(df_dist_mean.index,df_dist_mean.Eucl)
axe.axvspan(date_range[0], date_range[1], alpha=0.7, color='orange')

#Mahalanobis distance

df_dist_mean['MD'] = MahalanobisDist(inv_cov, mean_distr, data)

fig = plt.figure(figsize = (15,8))
axe = fig.add_axes([0.1,0.1,0.8,0.8])
axe.set_title('Mahalanobis distance')
axe.plot(df_dist_mean.index,df_dist_mean.MD)
axe.axvspan(date_range[0], date_range[1], alpha=0.7, color='orange')


#%% with standardised values

#standardise dataframe
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
data_x = min_max_scaler.fit_transform(data)

df_dist_mean_x = pd.DataFrame(data_x,index = df_dist_mean.index, columns = df_dist_mean.columns[:-2])
df_train_x = df_dist_mean_x[df_dist_mean_x.index.day == 13]

data_train_x = np.array(df_train_x.values)

cov_x,inv_cov_x = cov_matrix(data_train_x)
mean_distr_x = data_train_x.mean(axis = 0)

#Euclidean distance scaler (x)
df_dist_mean_x['Eucl'] = EuclideanDist(mean_distr_x, data_x)

fig = plt.figure(figsize = (15,8))
axe = fig.add_axes([0.1,0.1,0.8,0.8])
axe.set_title('L2 normal distance - Standardised')
axe.plot(df_dist_mean_x.index,df_dist_mean_x.Eucl)
axe.axvspan(date_range[0], date_range[1], alpha=0.7, color='orange')

#Mahalanobis distance scaled (x)

df_dist_mean_x['MD'] = MahalanobisDist(inv_cov_x, mean_distr_x, data_x)

fig = plt.figure(figsize = (15,8))
axe = fig.add_axes([0.1,0.1,0.8,0.8])
axe.set_title('Mahalanobis distance - Standardised')
axe.plot(df_dist_mean_x.index,df_dist_mean_x.MD)
axe.axvspan(date_range[0], date_range[1], alpha=0.7, color='orange')