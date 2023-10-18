# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 12:41:19 2023

Method plot to show correct window with centred smoothing

@author: George
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from datetime import timedelta
from statsmodels.tsa.api import SimpleExpSmoothing
from statsmodels.tsa.api import Holt
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

def get_means(datasets):
    
    mean = []
    for data in datasets:
        
        mean_data = {s:None for s in data.species}
        for s in mean_data:
            mean_data[s] = data.data_short[s].mean(axis = 1)
            
        mean.append(mean_data)
        
    return mean

def get_IGTs(datasets):
    
    IGT = []
    for data in datasets:
        
        IGT_data = {s:None for s in data.species}
        for s in IGT_data:
            IGT_data[s] = data.data_short[s].quantile(0.129167,axis = 1)
            
        IGT.append(IGT_data)
        
    return IGT


if __name__ == '__main__':
    
    #print
    dope_df = dope_read_extend()
    datas = [csvDATA(root,dope_df) for root in training_sets]
    means = get_means(datas)
    IGTs = get_IGTs(datas)
    
    #methods
    methods = [
        'Centred Rolling',
        'Kernel Gaussian Centred',
        'Kernel Exponential',
        'Exponential Single',
        'Exponential Double']
    
    #%%plot data
    directory = r'C:\Users\George\Documents\Figures\Methods'
    timestep = 5 #rollingmean
    t = timestep * 3
    alpha = t/8
    stdev = (t-1)/(2*alpha)
    tau = 3
    a,b = 0.15,0.07
    
    
    plt.close('all')
    fig_m,axes_m = plt.subplots(4,3,figsize = (19,22),sharex = True)
    fig_i,axes_i = plt.subplots(4,3,figsize = (19,22),sharex = True)
    
    for row,mean_data in enumerate(means):
        for i,s in enumerate(mean_data):
            
            #calculate different means
            mean = mean_data[s].rolling(t,center = True).mean().dropna()
            mean_gauss = mean_data[s].rolling(window = t,win_type = 'gaussian',center = True).mean(std = stdev).dropna()
            mean_exp = mean_data[s].rolling(window=t,win_type = 'exponential',center = True).mean(tau = tau).dropna()
            smoother_m = SimpleExpSmoothing(mean_data[s])
            mean_exp_s = smoother_m.fit(smoothing_level=a,optimized = False).fittedvalues
            smoother_m = Holt(mean_data[s])
            mean_exp_d = smoother_m.fit(smoothing_level=a,smoothing_trend=b,optimized = False).fittedvalues
            
            #plot all means
            axes_m[row,i].plot(mean_data[s].index[150:300]/60,mean_data[s].values[150:300],color = datas[0].species_colors[s])
            axes_m[row,i].plot(mean.index[150:300]/60,mean.values[150:300],color = datas[0].colors[0],label = methods[0])
            axes_m[row,i].plot(mean_gauss.index[150:300]/60,mean_gauss.values[150:300],color = datas[0].colors[1],label = methods[1])
            axes_m[row,i].plot(mean_exp.index[150:300]/60,mean_exp.values[150:300],color = datas[0].colors[2],label = methods[2]) #Exponential window
            axes_m[row,i].plot(mean_exp_s.index[150:300]/60,mean_exp_s.values[150:300],color = datas[0].colors[3],label = methods[3]) #Exponential single
            axes_m[row,i].plot(mean_exp_d.index[150:300]/60,mean_exp_d.values[150:300],color = datas[0].colors[4],label = methods[4]) #Exponential double
            #dopage
            axes_m[row,i].axvline(0,color = 'black')
            
            if not i: axes_m[row,i].set_ylabel(datas[row].date)
            
    for row,IGT_data in enumerate(IGTs):
        for i,s in enumerate(IGT_data):            
            
            IGT = IGT_data[s].rolling(t,center = True).mean().dropna()
            IGT_gauss = IGT_data[s].rolling(window = t,win_type = 'gaussian',center = True).mean(std = stdev).dropna()
            IGT_exp = IGT_data[s].rolling(window=t,win_type = 'exponential',center = True).mean(tau = tau).dropna()
            smoother_i = SimpleExpSmoothing(IGT_data[s])
            IGT_exp_s = smoother_i.fit(smoothing_level=a,optimized = False).fittedvalues
            smoother_i = Holt(IGT_data[s])
            IGT_exp_d = smoother_i.fit(smoothing_level=a,smoothing_trend=b,optimized = False).fittedvalues
            
            axes_i[row,i].plot(IGT_data[s].index[150:300]/60,IGT_data[s].values[150:300],color = datas[0].species_colors[s])
            axes_i[row,i].plot(IGT.index[150:300]/60,IGT.values[150:300],color = datas[0].colors[0],label = 'Centred Rolling')
            axes_i[row,i].plot(IGT_gauss.index[150:300]/60,IGT_gauss.values[150:300],color = datas[0].colors[1],label = 'Kernel Gaussian Centred')
            axes_i[row,i].plot(IGT_exp.index[150:300]/60,IGT_exp.values[150:300],color = datas[0].colors[2],label = 'Kernel Exponential') #Exponential window
            axes_i[row,i].plot(IGT_exp_s.index[150:300]/60,IGT_exp_s.values[150:300],color = datas[0].colors[3],label = 'Exponential Single') #Exponential single
            axes_i[row,i].plot(IGT_exp_d.index[150:300]/60,IGT_exp_d.values[150:300],color = datas[0].colors[4],label = 'Exponential Double') #Exponential double
            #dopage
            axes_i[row,i].axvline(0,color = 'black')
            
            if not i: axes_i[row,i].set_ylabel(datas[row].date)
            
    handles, labels = axes_m[0,i].get_legend_handles_labels()
    fig_m.legend(handles, labels)
    fig_i.legend(handles, labels)
            
    for i,s in enumerate(list(datas[0].species.values())):
        axes_m[0,i].set_title(s)
        axes_i[0,i].set_title(s)
        
    fig_m.suptitle('Mean data smoothing')
    fig_i.suptitle('IGT data smoothing')
    
    #style choice
    fig_m,axes_m = plt.subplots(5,1,figsize = (17,35),sharex = True)
    fig_i,axes_i = plt.subplots(5,1,figsize = (17,35),sharex = True)
    
    mean_data,IGT_data = means[2],IGTs[2]
    s = 'R'
        
    mean = mean_data[s].rolling(t,center = True).mean().dropna()
    mean_gauss = mean_data[s].rolling(window = t,win_type = 'gaussian',center = True).mean(std = stdev).dropna()
    mean_exp = mean_data[s].rolling(window=t,win_type = 'exponential',center = True).mean(tau = tau).dropna()
    smoother_m = SimpleExpSmoothing(mean_data[s])
    mean_exp_s = smoother_m.fit(smoothing_level=a,optimized = False).fittedvalues
    smoother_m = Holt(mean_data[s])
    mean_exp_d = smoother_m.fit(smoothing_level=a,smoothing_trend=b,optimized = False).fittedvalues
    
    IGT = IGT_data[s].rolling(t,center = True).mean().dropna()
    IGT_gauss = IGT_data[s].rolling(window = t,win_type = 'gaussian',center = True).mean(std = stdev).dropna()
    IGT_exp = IGT_data[s].rolling(window=t,win_type = 'exponential',center = True).mean(tau = tau).dropna()
    smoother_i = SimpleExpSmoothing(IGT_data[s])
    IGT_exp_s = smoother_i.fit(smoothing_level=a,optimized = False).fittedvalues
    smoother_i = Holt(IGT_data[s])
    IGT_exp_d = smoother_i.fit(smoothing_level=a,smoothing_trend=b,optimized = False).fittedvalues
    
    
    #Normal
    for x,m in enumerate(methods):
        axes_m[x].plot(mean_data[s].index/60,mean_data[s].values,color = datas[0].species_colors[s])
        axes_m[x].axvline(0,color = 'black')
        
        axes_i[x].plot(IGT_data[s].index/60,IGT_data[s].values,color = datas[0].species_colors[s])
        axes_i[x].axvline(0,color = 'black')
        
        axes_m[x].set_title('{}'.format(m))
        axes_i[x].set_title('{}'.format(m))
        
    #titles
    fig_m.suptitle('{} Means - Tox:{} {}'.format(datas[0].species[s],datas[0].Tox,datas[0].date))
    fig_i.suptitle('{} IGT - Tox:{} {}'.format(datas[0].species[s],datas[0].Tox,datas[0].date))
    
    #Moving mean
    axes_m[0].plot(mean.index/60,mean.values,color = 'black')
    axes_i[0].plot(IGT.index/60,IGT.values,color = 'black')
    
    #Gaussian
    axes_m[1].plot(mean_gauss.index/60,mean_gauss.values,color = 'black')
    axes_i[1].plot(IGT_gauss.index/60,IGT_gauss.values,color = 'black')
    
    #Exponential kernel
    axes_m[2].plot(mean_exp.index/60,mean_exp.values,color = 'black')
    axes_i[2].plot(IGT_exp.index/60,IGT_exp.values,color = 'black')
    
    #Exponential single
    axes_m[3].plot(mean_exp_s.index/60,mean_exp_s.values,color = 'black')
    axes_i[3].plot(IGT_exp_s.index/60,IGT_exp_s.values,color = 'black')
    
    #Exponential double
    axes_m[4].plot(mean_exp_d.index/60,mean_exp_d.values,color = 'black')
    axes_i[4].plot(IGT_exp_d.index/60,IGT_exp_d.values,color = 'black')