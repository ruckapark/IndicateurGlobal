# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 11:51:51 2023

Compare results of datafiles from the replay versions of files

@author: George
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from datetime import timedelta
from scipy import signal

#%% IMPORT personal mods
os.chdir('MODS')
from data_merge import merge_dfs
from dope_reg import dope_read_extend
import dataread as d_
os.chdir('..')

#%% Relevant directories
roots = ['765_20211022',
         '767_20211022',
         '762_20211022',
         '763-20211022',
         '762_20211028',
         '762_20220225',
         '763_20220225',
         '764_20220310',
         '765_20220310',
         '765_20220317']

def TLCC(df1,df2):
    """ Time lagged cross correlation - expecting high positive value """
    arg_lags = np.zeros(df1.shape[1])
    
    for x,col in enumerate(df1.columns):
    
        i = np.random.randint(50,500)
        
        #what if dead organism (error?)
        while True:
            serie1 = df1[col].iloc[i:i+30]
            if np.sum(serie1<10) > 10: #only take a series where there is significant movement
                i += 100
            else:
                break
        serie2 = df2[col].iloc[i:i+30]
        if np.sum(serie2<10) > 10: continue
        
        c = np.correlate(serie1, serie2, 'full')
        lags = signal.correlation_lags(len(serie1), len(serie2))
        arg_lags[x] = lags[np.argmax(c)]
    
    return np.median(arg_lags)

def plot_distribution(val1,val2,species = 'R'):
    xlims = {'E':350,'G':350,'R':200}
    
    plt.figure()
    sns.histplot(val1)
    sns.histplot(val2)
    plt.xlim(0,xlims[species])
    
    plt.figure()
    plt.plot(np.arange(0,1,0.01),np.array([np.quantile(val1,i/100) for i in range(100)]))
    plt.plot(np.arange(0,1,0.01),np.array([np.quantile(val2,i/100) for i in range(100)]))


#%% Code

"""
% of time higher then the other
average difference between values 
"""

if __name__ == '__main__':
    
    specie = {'E': 'Erpobdella','G':'Gammarus','R':'Radix'}
    
    root = roots[0]
    Tox = root.split('_')[0]
    
    stem = [r for r in os.listdir(r'I:\TXM{}-PC'.format(Tox)) if root.split('_')[-1] in r]
    root = r'I:\TXM{}-PC\{}'.format(Tox,stem[0])
    
    #locate original and copy
    file_og = r'{}\{}.xls.zip'.format(root,stem[0])
    file_copy = r'{}\{}.replay.xls.zip'.format(root,stem[0])
    
    #read file
    df_og,df_copy = d_.read_merge([file_og]),d_.read_merge([file_copy])
    dfs_og,dfs_copy = d_.preproc(df_og),d_.preproc(df_copy)
    
    #register
    #dope_df = dope_read_extend()
        
        
    #%% Use Gammarus for lag estimates (TLCC)
    df1,df2 = dfs_og['G'],dfs_copy['G']
    indexing = min(df1.shape[0],df2.shape[0])
    df1,df2 = df1.iloc[:indexing],df2.iloc[:indexing]
    
    tlc = TLCC(df1,df2)
    print("Median lag between series: ",tlc)
    if tlc != 0: print('tracking lag error!')
    
    
    #%% Start with Radix
    species = 'R'
    df1,df2 = dfs_og[species],dfs_copy[species]
    indexing = min(df1.shape[0],df2.shape[0])
    df1,df2 = df1.iloc[:indexing],df2.iloc[:indexing]
    df1_m,df2_m = d_.rolling_mean(df1,5),d_.rolling_mean(df2,5)
    
    fig,axe = plt.subplots(2,8,figsize = (25,6),sharex = True)
    for i in range(16):
        axe[i//8,i%8].plot(np.arange(df1.shape[0]),df1[i+1])
        axe[i//8,i%8].plot(np.arange(df2.shape[0]),df2[i+1])
    fig.tight_layout()
    
    fig,axe = plt.subplots(2,8,figsize = (25,6),sharex = True,sharey = True)
    for i in range(16):
        axe[i//8,i%8].plot(np.arange(df1_m.shape[0]),df1_m[i+1])
        axe[i//8,i%8].plot(np.arange(df2_m.shape[0]),df2_m[i+1])
        axe[i//8,i%8].set_ylim([0,200])
    fig.tight_layout()
    
    #plot distribution comparison
    values1,values2 = df1.values.flatten(),df2.values.flatten()
    values1,values2 = values1[values1 > 0],values2[values2 > 0]
    
    plot_distribution(values1,values2,species)
    
    #%% Erpobdella
    species = 'E'
    df1,df2 = dfs_og[species],dfs_copy[species]
    indexing = min(df1.shape[0],df2.shape[0])
    df1,df2 = df1.iloc[:indexing],df2.iloc[:indexing]
    df1_m,df2_m = d_.rolling_mean(df1,5),d_.rolling_mean(df2,5)
    
    fig,axe = plt.subplots(2,8,figsize = (25,6),sharex = True)
    for i in range(16):
        axe[i//8,i%8].plot(np.arange(df1.shape[0]),df1[i+1])
        axe[i//8,i%8].plot(np.arange(df2.shape[0]),df2[i+1])
    fig.tight_layout()
    
    fig,axe = plt.subplots(2,8,figsize = (25,6),sharex = True,sharey = True)
    for i in range(16):
        axe[i//8,i%8].plot(np.arange(df1_m.shape[0]),df1_m[i+1])
        axe[i//8,i%8].plot(np.arange(df2_m.shape[0]),df2_m[i+1])
        axe[i//8,i%8].set_ylim([0,350])
    fig.tight_layout()
    
    #plot distribution comparison
    values1,values2 = df1.values.flatten(),df2.values.flatten()
    values1,values2 = values1[values1 > 0],values2[values2 > 0]
    
    plot_distribution(values1,values2,species)