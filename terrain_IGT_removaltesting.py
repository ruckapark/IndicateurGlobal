# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 09:35:01 2021

Check performance of new IGT for random removal of organismes 16 (full) -> 15 -> 14 ... -> 1

@author: Admin
"""


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import dataread_terrain as d_terr
from datetime import timedelta
from data_merge import merge_dfs

plt.close('all')

def select_values(m,means):
    
    """
    Return the individuals to look at when simulating death in the system
    """
    
    if m == 1:
        return means.iloc[[16//2]].index
    elif m == 2:
        return means.iloc[[16//3,2*16//3]].index
    elif m == 3:
        return means.iloc[[16//4,2*16//4,3*16//4]].index
    elif m == 4:
        return means.iloc[[16//5,2*16//5,3*16//5,4*16//5]].index
    elif m == 5:
        return means.iloc[[16//5,2*16//5,3*16//5,4*16//5,5*(16//5)]].index
    elif m == 6:
        return means.iloc[[2,4,7,9,12,14]].index
    elif m == 7:
        return means.iloc[[2,4,7,8,9,12,14]].index
    elif m == 8:
        return means.iloc[[1,3,5,7,9,11,13,15]].index
    elif m == 9:
        return means.iloc[[1,3,5,7,8,9,11,13,15]].index
    elif m == 10:
        return means.iloc[[1,3,5,7,8,9,11,13,14,15]].index
    elif m == 11:
        return means.iloc[[0,1,3,5,7,8,9,11,13,14,15]].index
    elif m == 12:
        return means.iloc[[0,1,3,5,7,8,9,10,11,13,14,15]].index
    elif m == 13:
        return means.iloc[[0,1,2,3,5,7,8,9,10,11,13,14,15]].index
    elif m == 14:
        return means.iloc[[0,1,2,3,5,7,8,9,10,11,12,13,14,15]].index
    elif m == 15:
        return means.iloc[[0,1,2,3,4,5,7,8,9,10,11,12,13,14,15]].index
    else:
        return None



if __name__ == '__main__':

    colors = [
        '#42f5e0','#12aefc','#1612fc','#6a00a3',
        '#8ef743','#3c8f01','#0a4001','#fc03ca',
        '#d9d200','#d96c00','#942c00','#fc2803',
        '#e089b6','#a3a3a3','#7a7a7a','#303030'
        ]
    
    root = r'D:\VP\Viewpoint_data\Suez'
    os.chdir(root)
    files = [f for f in os.listdir() if '02.csv' in f]
    
    
    print('The following files will be studied:')
    print(files)
    
    dfs,dfs_mean = d_terr.read_data_terrain(files)
    
    # extract relevant dfs
    species = 'G'
    df = dfs[species]
    df_mean = dfs_mean[species]
    
    #mean values to be used for gradually removing individuals
    means = df.mean().sort_values()
    
    # plot individually (plot16)
    fig,axe = d_terr.plot_16(df)
    
    """
    Check effect on percentage if alive
    
    #number of morts
    for m in range(16):
        
        if m: 
            df_adj = df.drop(columns = select_values(m,means))
        else:
            df_adj = df.copy()
        
        IGT_old  = df_adj.quantile(q = 0.05,axis = 1)**2
        
        #values in np array form
        values = np.array(df_adj)
        values.sort()
        
        coeffs = sigmoid_coeffs(m,species)
        
        IGT = np.sum(values**coeffs,axis = 1)
    
        fig,axe = plt.subplots(3,1,sharex = True,figsize = (18,9))
        axe[0].plot(df.index,IGT,color = 'b')
        axe[1].plot(df.index,IGT_old,color = 'r')
        axe[2].plot(df.index,percent_IGT(IGT, species))
        plt.suptitle('Morts: {}'.format(m))
        
    """
    
    #here there is no mortality
    m = 0
    
    #values in np array form
    values = np.array(df)
    values.sort()
    
    coeffs = d_terr.sigmoid_coeffs(int(16*m),species)
    IGT = np.sum(values**coeffs,axis = 1)
    IGT_ = d_terr.percent_IGT(IGT, species)
    fig,axe = plt.subplots(1,1,figsize = (18,9))
    axe.plot(df.index,IGT,color = 'green')
    