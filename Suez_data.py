# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 11:17:28 2021

Suez data generation

@author: Admin
"""


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import dataread_terrain as data_
from remove_organism_terrain2 import search_dead

### Main

plt.close('all')

colors = [
    '#42f5e0','#12aefc','#1612fc','#6a00a3',
    '#8ef743','#3c8f01','#0a4001','#fc03ca',
    '#d9d200','#d96c00','#942c00','#fc2803',
    '#e089b6','#a3a3a3','#7a7a7a','#303030'
    ]

root = r'D:\VP\Viewpoint_data\Suez'
os.chdir(root)
files = [f for f in os.listdir() if '.csv' in f]

print('The following files will be studied:')
print(files)

for file in files:

    dfs,dfs_mean = data_.read_data_terrain([file])  
    
    for species in ['E','G','R']:
        
        df = dfs[species]
        df_mean = dfs_mean[species]
        
        # plot all on same figure - no mean and mean
        data_.single_plot(df,species,title = 'Distance covered')
        data_.single_plot(df_mean,species,title = 'Distance covered movingmean')
        
        # plot individually (plot16)
        fig,axe = data_.plot_16(df)
        
        ## Main
        
        # Caluclate deaths
        data = np.array(df)
        data_alive,data_counters = search_dead(data,species)
                
        #form pandas
        df_alive = pd.DataFrame(data_alive, index = df.index, columns = df.columns)
        df_counters = pd.DataFrame(data_counters, index = df.index, columns = df.columns)
        
        #plot 
        fig,axe = data_.plot_16(df_alive)
        fig,axe = data_.plot_16(df_counters)
        
        #mins = df[df_alive == 1].min(axis = 1)**2
        IGT = df[df_alive == 1].quantile(0.15,axis = 1)**1.75
        if species == 'E':
            IGT[IGT > 5000] = 5000
        elif species == 'G':
            IGT[IGT > 10000] = 10000
        elif species == 'R':
            IGT[IGT > 1200] = 1200
        else:
            break
            
        mortality = 1 - df_alive.sum(axis = 1)/16
        
        fig,ax1 = plt.subplots(figsize = (13,8))
        fig.suptitle(species)
        #ax1.plot(mins.index,mins,color = 'blue')
        ax1.plot(IGT.index,IGT,color = 'red')
        ax1.set_ylabel('IGT')
        for tick in ax1.get_xticklabels():
            tick.set_rotation(90)
        
        ax2 = ax1.twinx()
        ax2.plot(mortality.index,mortality,color = 'orange')
        ax2.set_ylabel('Mortalite')
        
        results = pd.DataFrame(columns = ['toxicityindex'],index = IGT.index)
        results.index.names = ['time']
        results['toxicityindex'] = IGT
        #results['Mortalite'] = mortality
        results.to_csv('{}data_{}.csv'.format(file.split('.')[0],species))