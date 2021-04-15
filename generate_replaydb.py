# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 15:25:27 2021

@author: Admin
"""
    


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dataread_terrain as d_terr

if __name__ == '__main__':

    colors = [
        '#42f5e0','#12aefc','#1612fc','#6a00a3',
        '#8ef743','#3c8f01','#0a4001','#fc03ca',
        '#d9d200','#d96c00','#942c00','#fc2803',
        '#e089b6','#a3a3a3','#7a7a7a','#303030'
        ]
    
    root = r'D:\VP\Viewpoint_data\Suez'
    os.chdir(root)
    files = [f for f in os.listdir()]
    
    print('The following files will be studied:')
    print(files)
    
    for file in files:
    
        dfs,dfs_mean = d_terr.read_data_terrain([file])
        
        # extract relevant dfs
        for species in 'G E R'.split():
        
            df = dfs[species]
            df_mean = dfs_mean[species]
            
            # plot individuals
            fig,axe = d_terr.plot_16(df)
            
            # find deaths
            data_alive,data_counters = d_terr.search_dead(np.array(df),species)
            
            # hide flaw in ToxMate...
            if ('1402.csv' in file) & (species == 'G'):
                data_alive = np.ones_like(data_alive)
            
            #mortality percentage in time
            m = np.ones(len(data_alive),dtype = float) - (np.sum(data_alive,axis = 1))/16
            
            #dist np array
            values = np.array(df)
            
            #remove values mort
            values[data_alive == 0] = np.nan
            
            # values[i][0] < values[i][1] < values[i][2]
            values.sort()
            
            IGT = np.zeros_like(m)
            old_IGT = np.zeros_like(m)
            for i in range(len(values)):
                coeffs = d_terr.sigmoid_coeffs(m[i],species)
                IGT[i] = np.sum(values[i][:len(coeffs)]**coeffs)
                #check if all values nan (100% mortality)
                if np.isnan(values[i][0]):
                    old_IGT[i] = 0
                else:
                    old_IGT[i] = np.quantile(values[i][~np.isnan(values[i])],0.05)**2
               
            # caluclate IGT from raw -> %    
            IGT = d_terr.percent_IGT(IGT, species)
            
            #compare old and new values
            fig,axe = plt.subplots(2,1,figsize = (18,9),sharex = True)
            plt.suptitle('IGT 5% vs. percent new_IGT')
            axe[0].plot(df.index,old_IGT,color = 'green')
            axe[1].plot(df.index,IGT,color = 'green')
            axe[1].tick_params(axis='x', rotation=90)
            
            
            res = d_terr.save_results(df.index, IGT, m, species, file)
            #save to txt files
            #d_terr.gen_txt(res,file,species)
            
    # merge results to file for replaydb
    # d_terr.join_text()...