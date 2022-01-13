# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 16:34:34 2021

Add limit to death in systems

@author: George
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%% IMPORT personal mods
os.chdir('MODS')
import dataread_terrain as d_terr
os.chdir('..')
import TERRAIN_readdata as d

#%% Functions

def living_organisms(df):
    cols = []
    for i in range(df.shape[0]):
        cols.append(list(df.iloc[i][df.iloc[i].notnull()].index))
    return cols

def prob_dist(lim):
    base_dist = [1 - 0.05*x for x in range(10)]
    return base_dist[:lim]

root = r'D:\VP\Viewpoint_data\TERRAIN\Suez'
files = ['toxmate_0708_240821.csv']
data,thresh = d.main(files,'EGR',root = root)

#%%COD3

#%%

species = 'R'
df = data[species]['df_m']
living_cols = living_organisms(df)
trueIGT = pd.DataFrame(data[species]['IGT'],index = df.index)


#%% test for 5 organisms
results = {x:pd.DataFrame() for x in range(1,8)}
for day in range(8,24):
    
    df_day = df[df.index.day == day]
    day_IGT = np.array(trueIGT[trueIGT.index.day == day])
    cols = np.array(df_day.isnull().sum().sort_values().index)[:10]
    
    #number of alive organisms loop
    for j in range(1,8):
    
        score = np.zeros(33) #should be a normal distribution
        for i in range(33):
            subcols = np.random.choice(cols,j,prob_dist(j))
            subdf = np.array(df_day[subcols])
            IGT_sim = np.zeros(len(subdf))
            for x in range(len(IGT_sim)):
                IGT_sim[x] = d_terr.IGT_(subdf[x],species,thresh)
               
            """
            # plt.figure()
            # plt.plot(IGTtemp)
            # plt.plot(temp_trueIGT,'r')
            """
            
            scores = pd.DataFrame(index = df_day.index)
            scores['IGT_sim'] = IGT_sim
            scores['IGT_true'] = day_IGT
            scores['diffs'] = scores['IGT_sim'] - scores['IGT_true']
            score[i] = (len(scores[(scores['IGT_sim']>75)&(scores['IGT_true']<50)]))
            
        results[j][day] = score
        #print('alive = {}, score = {}'.format(j,np.median(np.array(score))))

#%% plot distribution of results across days and 33 random selections        
for i in range(1,8):
    plt.figure()
    sns.histplot(np.array(results[i]).flatten()/(len(IGT_sim)/100))
    plt.title('Score {} {}organismes'.format(species,i))
    plt.xlabel('Percentage of false corrupt data')
    print(i,' orgainisms median: ',np.median(np.array(results[i]).flatten())/7.2,'mean: ',np.mean(np.array(results[i]).flatten())/7.2)