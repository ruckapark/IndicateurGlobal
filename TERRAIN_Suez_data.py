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

#%% IMPORT personal mods
os.chdir('MODS')
import dataread_terrain as d_terr
os.chdir('..')

### Main

plt.close('all')

colors = [
    '#42f5e0','#12aefc','#1612fc','#6a00a3',
    '#8ef743','#3c8f01','#0a4001','#fc03ca',
    '#d9d200','#d96c00','#942c00','#fc2803',
    '#e089b6','#a3a3a3','#7a7a7a','#303030'
    ]

def main(filename):
    root = r'D:\VP\Viewpoint_data\Suez'
    os.chdir(root)
    files = filename
    
    print('The following files will be studied:')
    print(files)
    
    for file in files:
    
        dfs,dfs_mean = d_terr.read_data_terrain([file])  
        
        for species in ['E','G','R']:
            
            df = dfs[species]
            df_mean = dfs_mean[species]
            
            # plot all on same figure - no mean and mean
            d_terr.single_plot(df,species,title = 'Distance covered')
            d_terr.single_plot(df_mean,species,title = 'Distance covered movingmean')
            
            # plot individually (plot16)
            fig,axe = d_terr.plot_16(df)
            
            ## Main
            
            # Caluclate deaths
            data = np.array(df)
            data_alive,data_counters = d_terr.search_dead(data,species)
                    
            #form pandas
            df_alive = pd.DataFrame(data_alive, index = df.index, columns = df.columns)
            df_counters = pd.DataFrame(data_counters, index = df.index, columns = df.columns)
            
            #plot 
            fig,axe = d_terr.plot_16(df_alive)
            fig,axe = d_terr.plot_16(df_counters)
            
            #old style IGT
            IGT = df[df_alive == 1].quantile(0.05,axis = 1)**2
                
            mortality = np.array(1 - df_alive.sum(axis = 1)/16)
            #new style IGT in percentage
            IGT_percent = d_terr.IGT_per(data,data_alive,mortality,species)
            
            fig,ax1 = plt.subplots(figsize = (13,8))
            fig.suptitle(species)
            ax1.plot(IGT.index,IGT,color = 'green')
            ax1.set_ylabel('IGT')
            for tick in ax1.get_xticklabels():
                tick.set_rotation(90)
            
            ax2 = ax1.twinx()
            ax2.plot(IGT.index,IGT_percent/100,color = 'blue')
            ax2.plot(IGT.index,mortality,color = 'orange')
            ax2.set_ylabel('Mortalite')
            
            results = pd.DataFrame(columns = ['toxicityindex'],index = IGT.index)
            results.index.names = ['time']
            results['toxicityindex'] = IGT
            #results['Mortality'] = mortality
            #results.to_csv('{}data_{}.csv'.format(file.split('.')[0],species))
            
            return IGT_percent
            
if __name__ == '__main__':
    #filename couuld be list too
    filename = [r'D:\VP\Viewpoint_data\Suez\toxmate_0102_1402.csv']
    test = main(filename)