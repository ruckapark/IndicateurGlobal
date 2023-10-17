# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 18:48:48 2023

@author: George
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from datetime import timedelta
from LAB_ToxClass import csvDATA

#%% IMPORT personal mods
os.chdir('MODS')
from data_merge import merge_dfs
from dope_reg import dope_read_extend
import dataread as d_
os.chdir('..')

#copper examples
training_sets = [
    r'I:\TXM765-PC\20210422-111620',
    r'I:\TXM767-PC\20210430-124553',
    r'I:\TXM767-PC\20210513-231929',
    r'I:\TXM763-PC\20210528-113951'
    ]

#methomyl examples
test_sets = [
    r'I:\TXM760-PC\20210520-224501',
    r'I:\TXM760-PC\20210625-093621',
    r'I:\TXM761-PC\20210520-224549',
    r'I:\TXM761-PC\20210625-093641'
    ]


if __name__ == '__main__':
    
    #print
    dope_df = dope_read_extend()
    data = csvDATA(training_sets[0],dope_df)
    
    mean_data = {s:None for s in data.species}
    for s in mean_data:
        mean_data[s] = data.data_short[s].mean(axis = 1)
        
    IGT_data = {s:None for s in data.species}
    for s in IGT_data:
        IGT_data[s] = data.data_short[s].quantile(0.129167,axis = 1)
    
    #%%plot data
    plt.close('all')
    fig_m,axes_m = plt.subplots(1,3,figsize = (19,7),sharex = True)
    fig_i,axes_i = plt.subplots(1,3,figsize = (19,7),sharex = True)
    
    for i,s in enumerate(mean_data):
        axes_m[i].plot(mean_data[s].index[:400],mean_data[s].values[:400],color = data.species_colors[s])
        axes_m[i].axvline(0,color = 'black')
        
    for i,s in enumerate(IGT_data):
        axes_i[i].plot(IGT_data[s].index[:400],IGT_data[s].values[:400],color = data.species_colors[s])
        axes_i[i].axvline(0,color = 'black')
        
    test = 'mean_rolling_righttrail'
        
    """
    Moving means
    d_.rolling_mean(df,5) #5 minutes used typically on rolling mean
    d_.rolling_mean(d_.block_mean(df,10), 4) #grouped mean + rolling mean used in article 1
    
    Rolling mean function: default right edge mean - centre may be better
    
    Here we will try both and more
    """
    if test == 'mean_rolling_righttrail':
        
        spacing  = np.array([2,5,10,15,20,30])
        timesteps = spacing * 3
        for x,t in enumerate(timesteps):
            
            for i,s in enumerate(mean_data):
                
                mean = mean_data[s].rolling(t).mean().dropna()
                IGT = IGT_data[s].rolling(t).mean().dropna()
                
                axes_m[i].plot(mean.index[:400-t],mean.values[:400-t],color = data.colors[x],label = '{} mins'.format(t//3))
                axes_i[i].plot(IGT.index[:400-t],IGT.values[:400-t],color = data.colors[x],label = '{} mins'.format(t//3))
                
        fig_m.suptitle('{} Mean data'.format(test))
        fig_i.suptitle('{} IGT data'.format(test))
        
        handles, labels = axes_m[i].get_legend_handles_labels()
        fig_m.legend(handles, labels)
        fig_i.legend(handles, labels)
    
    
    """
    Gaussian means
    
    """
    
    """
    #Exponent means
    
    """