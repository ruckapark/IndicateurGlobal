# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 10:10:14 2021

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

#developped in optimisation
seuil_bdf = {'G':[0.7,19],'E':[0.7,18],'R':[0.8,5]}
cutoff = {'G':[2000,3500,12000],'E':[1000,2500,10000],'R':[250,450,1200]}
offsets = {'G':3120,'E':1869,'R':406} #parametres optimises pour cannes

def bruit_de_fond(values,species):
    """ 
    Quantile bruit de fond
    Facteur division
    """
    seuil = seuil_bdf[species]
    # quantile / facteur
    bdf = np.nanquantile(values,seuil[0])/seuil[1]
    if np.isnan(bdf):
        bdf = 0
    return bdf

def IGT_percent(IGT,species):
    
    """
    Transform IGT -> percent
    """
    #si Gammarus - seuil = [2000,3500,1200]
    seuil  = cutoff[species]
    offset = offsets[species]
    
    if IGT < seuil[0]:
        return (IGT / (seuil[0]/45))
    elif IGT < seuil[1]:
        return ((IGT - seuil[0]) / ((seuil[1] - seuil[0])/25)) + 45
    else:
        return (np.log(IGT - offset) - np.log(seuil[1] - offset)) * (20 / np.log((seuil[2] - offset)/(seuil[1]-seuil[0]))) + 70 
    


### Main

if __name__ == '__main__':

    colors = [
        '#42f5e0','#12aefc','#1612fc','#6a00a3',
        '#8ef743','#3c8f01','#0a4001','#fc03ca',
        '#d9d200','#d96c00','#942c00','#fc2803',
        '#e089b6','#a3a3a3','#7a7a7a','#303030'
        ]
    
    root = r'D:\VP\Viewpoint_data\Suez'
    os.chdir(root)
    files = [f for f in os.listdir() if '1802_1703.csv' in f]
    
    print('The following files will be studied:')
    print(files)
    
    dfs,dfs_mean = d_terr.read_data_terrain(files)
    
    #%%
    
    species = 'G'
    df = dfs[species]
    df_mean = dfs_mean[species]
    
    
    # plot all on same figure - no mean and mean
    d_terr.single_plot(df,species,title = 'Distance covered')
    d_terr.single_plot(df_mean,species,title = 'Distance covered movingmean')
    
    # plot individually (plot16)
    fig,axe = d_terr.plot_16(df_mean)    
        
    # means
    plt.figure()
    plt.plot(df.index,df.mean(axis = 1))
    
    #find the distribution of all the columns
    values = np.array(df)
    values.sort()
    
    #search data for organisms
    data_alive,data_counters = d_terr.search_dead(np.array(df),species)
    m = np.ones(len(data_alive),dtype = float) - (np.sum(data_alive,axis = 1))/16
    
    #find all values in csv (array [timebins,16])
    data = np.array(df)
    data[data_alive == 0] = np.nan
    
    #preallocate space
    IGT_ = np.zeros_like(m)
    IGT_per = np.zeros_like(m)
    
    # data (death correction) treatment
    # values[t] = 16 valeurs pour chaque timebin
    for t in range(len(data)):
        
        x = data[t]                 #timebin datum taille [16,]
        IGT = np.nanquantile(x,0.05)**2 #IGT old
        if np.isnan(IGT): IGT = 0   #remove nan
        
        bdf = bruit_de_fond(x, species)
        percent = IGT_percent(IGT,species)
        
        IGT_[t] = IGT
        IGT_per[t] = bdf + percent
        
        
    fig,axe = plt.subplots(2,1,sharex = True,figsize = (20,10))
    axe[0].plot(df.index,IGT_)
    axe[1].plot(df.index,IGT_per)