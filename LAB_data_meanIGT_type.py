# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 15:45:56 20

Compare mean from smoothed individual plots

Compare IGT from

@author: George
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from datetime import timedelta
from LAB_ToxClass import csvDATA,ToxPLOT

#%% IMPORT personal mods
os.chdir('MODS')
from data_merge import merge_dfs
from dope_reg import dope_read_extend
import dataread as d_
os.chdir('..')

def gaussian_smooth(data,timestep = 5,alpha = 8):
    
    timestep  = 5
    t = timestep * 3
    a = t/alpha
    stdev = (t-1)/(2*a)
    
    return data.rolling(window=t,win_type = 'gaussian',center = True).mean(std = stdev).dropna()

#copper examples
training_sets = [
    r'I:\TXM765-PC\20210422-111620',
    r'I:\TXM767-PC\20210430-124553',
    r'I:\TXM767-PC\20210513-231929',
    r'I:\TXM763-PC\20210528-113951'
    ]

if __name__ == '__main__':
    
    #print
    dope_df = dope_read_extend()
    data = csvDATA(training_sets[0],dope_df)
    
    df_raw,df_raw_s = {s:None for s in data.species},{s:None for s in data.species}
    mean_raw,mean_raw_s = {s:None for s in data.species},{s:None for s in data.species}
    IGT_raw,IGT_raw_s = {s:None for s in data.species},{s:None for s in data.species}
    mean_s,IGT_s = {s:None for s in data.species},{s:None for s in data.species}
    
    #compare mean smoothed vs mean of smoothed
    for s in data.species:
    
        df_raw[s] = data.data_short[s]
        df_raw_s[s] = gaussian_smooth(df_raw[s])
        
        mean_raw[s] = df_raw[s].mean(axis = 1)
        IGT_raw[s] = df_raw[s].quantile(0.129167,axis = 1)
        
        mean_raw_s[s] = gaussian_smooth(mean_raw[s])
        IGT_raw_s[s] = gaussian_smooth(IGT_raw[s])
        
        mean_s[s] = df_raw_s[s].mean(axis = 1)
        IGT_s[s] = df_raw_s[s].quantile(0.129167,axis = 1)
        
    #%% Plots
    plt.close('all')
    
    ToxPLOT(data).plot16('G',with_mean = True) #with mean true by default
    
    fig,axes = plt.subplots(2,3,figsize = (13,7),sharex = True)
    
    for i,s in enumerate(data.species):
    
        axes[0,i].axvline(0,color = 'black')
        axes[1,i].axvline(0,color = 'black')
        
        axes[0,i].plot(mean_raw[s].index,mean_raw[s].values,color = data.species_colors[s])
        axes[1,i].plot(IGT_raw[s].index,IGT_raw[s].values,color = data.species_colors[s])
        
        axes[0,i].plot(mean_raw_s[s].index,mean_raw_s[s].values,color = 'black',label = 'Filtered Raw Metric')
        axes[1,i].plot(IGT_raw_s[s].index,IGT_raw_s[s].values,color = 'black')
        
        axes[0,i].plot(mean_s[s].index,mean_s[s].values,color = 'red',label = 'Metric of Prefiltered Data')
        axes[1,i].plot(IGT_s[s].index,IGT_s[s].values,color = 'red')
    
    
    handles, labels = axes[0,i].get_legend_handles_labels()
    fig.legend(handles, labels)