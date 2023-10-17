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
    directory = r'C:\Users\George\Documents\Figures\Methods'
    
    plt.close('all')
    fig_m,axes_m = plt.subplots(1,3,figsize = (19,7),sharex = True)
    fig_i,axes_i = plt.subplots(1,3,figsize = (19,7),sharex = True)
    
    for i,s in enumerate(mean_data):
        axes_m[i].plot(mean_data[s].index[:400],mean_data[s].values[:400],color = data.species_colors[s])
        axes_m[i].axvline(0,color = 'black')
        
    for i,s in enumerate(IGT_data):
        axes_i[i].plot(IGT_data[s].index[:400],IGT_data[s].values[:400],color = data.species_colors[s])
        axes_i[i].axvline(0,color = 'black')
        
    test = 'Rolling mean RightTrail'
    test = 'Rolling mean Centre'
    test = 'Kernel Gaussian'
    #test = 'Kernel DPSS'
    test = 'Kernel Exponential'
        
    """
    Moving means
    d_.rolling_mean(df,5) #5 minutes used typically on rolling mean
    d_.rolling_mean(d_.block_mean(df,10), 4) #grouped mean + rolling mean used in article 1
    
    Rolling mean function: default right edge mean - centre may be better
    
    Here we will try both and more
    
    Various windows
    
    Chosen windows:
        -Gaussian
        -dpss - varied tail steepness
        -Exponential - steep centre based on tau  
    """
    if test == 'Rolling mean RightTrail':
        
        spacing  = np.array([3,4,5,6])
        timesteps = spacing * 3
        for x,t in enumerate(timesteps):
            
            for i,s in enumerate(mean_data):
                
                mean = mean_data[s].rolling(t).mean().dropna()
                IGT = IGT_data[s].rolling(t).mean().dropna()
                
                axes_m[i].plot(mean.index[:400-t],mean.values[:400-t],color = data.colors[x],label = '{} mins'.format(t//3))
                axes_i[i].plot(IGT.index[:400-t],IGT.values[:400-t],color = data.colors[x],label = '{} mins'.format(t//3))
                
        fig_m.suptitle('{} Mean data'.format(test))
        fig_i.suptitle('{} IGT data'.format(test))
        
        for i,s in enumerate(list(data.species.values())):
            axes_m[i].set_title(s)
            axes_i[i].set_title(s)
        
        handles, labels = axes_m[i].get_legend_handles_labels()
        fig_m.legend(handles, labels)
        fig_i.legend(handles, labels)
        
    elif test == 'Rolling mean Centre':
        
        spacing  = np.array([3,4,5,6])
        timesteps = spacing * 3
        for x,t in enumerate(timesteps):
            
            for i,s in enumerate(mean_data):
                
                mean = mean_data[s].rolling(t,center = True).mean().dropna()
                IGT = IGT_data[s].rolling(t,center = True).mean().dropna()
                
                axes_m[i].plot(mean.index[:400-t//2],mean.values[:400-t//2],color = data.colors[x],label = '{} mins'.format(t//3))
                axes_i[i].plot(IGT.index[:400-t//2],IGT.values[:400-t//2],color = data.colors[x],label = '{} mins'.format(t//3))
                
        fig_m.suptitle('{} Mean data'.format(test))
        fig_i.suptitle('{} IGT data'.format(test))
        
        for i,s in enumerate(list(data.species.values())):
            axes_m[i].set_title(s)
            axes_i[i].set_title(s)
        
        handles, labels = axes_m[i].get_legend_handles_labels()
        fig_m.legend(handles, labels)
        fig_i.legend(handles, labels)
    
    elif test == 'Kernel Gaussian':
        
        spacing  = 5
        t = spacing * 3
        alphas = t/np.array([2,4,6,8])
        for x,a in enumerate(alphas):
            
            stdev = (t-1)/(2*a)
            
            for i,s in enumerate(mean_data):
                
                mean = mean_data[s].rolling(window=t,win_type = 'gaussian',center = True).mean(std = stdev).dropna()
                IGT = IGT_data[s].rolling(window=t,win_type = 'gaussian',center = True).mean(std = stdev).dropna()
                
                axes_m[i].plot(mean.index[t//2:400-t//2],mean.values[t//2:400-t//2],color = data.colors[x],label = 'Alpha {}'.format(a))
                axes_i[i].plot(IGT.index[t//2:400-t//2],IGT.values[t//2:400-t//2],color = data.colors[x],label = 'Alpha {}'.format(a))
                
        fig_m.suptitle('{} Mean data'.format(test))
        fig_i.suptitle('{} IGT data'.format(test))
        
        for i,s in enumerate(list(data.species.values())):
            axes_m[i].set_title(s)
            axes_i[i].set_title(s)
        
        handles, labels = axes_m[i].get_legend_handles_labels()
        fig_m.legend(handles, labels)
        fig_i.legend(handles, labels)
        
    elif test == 'Kernel Exponential':
        
        spacing  = 5
        t = spacing * 3
        taus = np.array([1,2,3,4,5])
        for x,tau in enumerate(taus):
            
            #stdev = (t-1)/(2*a)
            
            for i,s in enumerate(mean_data):
                
                mean = mean_data[s].rolling(window=t,win_type = 'exponential',center = True).mean(tau = tau).dropna()
                IGT = IGT_data[s].rolling(window=t,win_type = 'exponential',center = True).mean(tau = tau).dropna()
                
                axes_m[i].plot(mean.index[t//2:400-t//2],mean.values[t//2:400-t//2],color = data.colors[x],label = 'Tau {}'.format(tau))
                axes_i[i].plot(IGT.index[t//2:400-t//2],IGT.values[t//2:400-t//2],color = data.colors[x],label = 'Tau {}'.format(tau))
                
        fig_m.suptitle('{} Mean data'.format(test))
        fig_i.suptitle('{} IGT data'.format(test))
        
        for i,s in enumerate(list(data.species.values())):
            axes_m[i].set_title(s)
            axes_i[i].set_title(s)
        
        handles, labels = axes_m[i].get_legend_handles_labels()
        fig_m.legend(handles, labels)
        fig_i.legend(handles, labels)
    
    
    """
    #Exponent means
    
    """