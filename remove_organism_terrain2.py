# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 15:39:34 2021

@author: Admin
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dataread_terrain as data_


##Functions

def search_dead(data,species):
    
    """
    Search for death in historical data    
    """
    
    #mins (*3 timestep 20s)
    threshold_death = {'E':180*3,'G':60*3,'R':120*3}
    thresh_life = {'G':4*3,'E':4*3,'R':8*3}
    
    data_alive = np.ones_like(data) 
    data_counters = np.zeros_like(data)
    
    # live storage
    counters = np.zeros(16)
    alive = np.ones(16)
    
    # through full dataset
    for i in range(thresh_life[species], len(data)):
        
        # through 16
        for x in range(len(data[0])):
            
            # if they are alive
            if alive[x]:
                
                if data[i][x]:
                    
                    if data[i-1][x]:
                        counters[x] = 0
                    else:
                        counters[x] += 1
                    
                else:
                    counters[x] += 1
            
            #if they are dead        
            else:
                if 0 not in data[(i- thresh_life[species]):i,x]:
                    alive[x] = 1
                    counters[x] = 0
                    
            if counters[x] >= threshold_death[species]:
                alive[x] = 0
                
            data_alive[i] = alive
            data_counters[i] = counters
            
    return data_alive,data_counters

##MAIN


colors = [
    '#42f5e0','#12aefc','#1612fc','#6a00a3',
    '#8ef743','#3c8f01','#0a4001','#fc03ca',
    '#d9d200','#d96c00','#942c00','#fc2803',
    '#e089b6','#a3a3a3','#7a7a7a','#303030'
    ]


#set working directory
root = r'D:\VP\Viewpoint_data\DATA_terrain'
os.chdir(root)
files = [f for f in os.listdir() if '.csv' in f]

#print working files
print('The following files will be studied:')
print(files)

#extract data and averaged data
dfs,dfs_mean = data_.read_data_terrain(files)


for species in ['E','G','R']:

    df = dfs[species]
    df_mean = dfs_mean[species]
    
    # plot all on same figure - no mean and mean
    data_.single_plot(df,species,title = 'Distance covered')
    data_.single_plot(df_mean,species,title = 'Distance covered movingmean')
    
    # plot individually (plot16)
    fig,axe = data_.plot_16(df_mean)
    
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
    
    mins = df[df_alive == 1].min(axis = 1)**2
    IGT = df[df_alive == 1].quantile(0.05,axis = 1)**2
    mortality = 1 - df_alive.sum(axis = 1)/16
    
    fig,ax1 = plt.subplots(figsize = (13,8))
    fig.suptitle(species)
    ax1.plot(mins.index,mins,color = 'blue')
    ax1.plot(IGT.index,IGT,color = 'red')
    ax1.set_ylabel('IGT')
    for tick in ax1.get_xticklabels():
        tick.set_rotation(90)
    
    ax2 = ax1.twinx()
    ax2.plot(mortality.index,mortality,color = 'orange')
    ax2.set_ylabel('Mortalite')