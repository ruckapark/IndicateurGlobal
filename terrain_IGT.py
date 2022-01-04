# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 15:33:30 2021

REWRITE FILE FOR ANY TESTING ON DEVELOPMENTS TO IMPROVE TERRAIN IGT

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
    

if __name__ == '__main__':
    
    root = r'D:\VP\Viewpoint_data\TERRAIN\Suez'
    os.chdir(root)
    files = ['toxmate_1005_1905.csv']
    
    print('The following files will be studied:')
    print(files)
    
    dfs,dfs_mean = d_terr.read_data_terrain(files)
    
    # extract relevant dfs
    for species in 'G E R'.split():
    
        df = dfs[species]
        df_mean = dfs_mean[species]
        
        # plot individually (plot16)
        fig,axe = d_terr.plot_16(df)
        
        # treat array for deaths 
        data_alive,data_counters = d_terr.search_dead(np.array(df),species)
        
        # m is list of mortality percentage
        m = np.ones(len(data_alive),dtype = float) - (np.sum(data_alive,axis = 1))/16
        
        
        #values in np array form
        values = np.array(df)
        
        #remove values mort
        values[data_alive == 0] = np.nan
        
        # values[i][0] < values[i][1] < values[i][2]
        values.sort()
        
        IGT = np.zeros_like(m)    
        for i in range(len(values)):
            coeffs = d_terr.sigmoid_coeffs(m[i],species)
            IGT[i] = np.sum(values[i][:len(coeffs)]**coeffs)
           
        # caluclate IGT from raw -> %    
        IGT = d_terr.percent_IGT(IGT, species)
        fig,axe = plt.subplots(1,1,figsize = (18,9))
        axe.plot(df.index,IGT,color = 'green')