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
from datetime import timedelta,datetime
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
         '763_20211022',
         '762_20211028',
         '762_20220225',
         '763_20220225',
         '764_20220310',
         '765_20220310',
         '765_20220317',
         '760_20220708',
         '761_20220708',
         '762_20220708',
         '763_20220708',
         '764_20220708',
         '765_20220708',
         '766_20220708',
         '767_20220708',
         '768_20220708',
         '769_20220708']

def plot_distribution(val1,val2,species = 'R',figname = None):
    xlims = {'E':1000,'G':1000,'R':200}
    
    plt.figure()
    sns.histplot(val1)
    sns.histplot(val2)
    plt.xlim(0,xlims[species])
    
    if figname: plt.savefig(r'C:\Users\George\Documents\Figures\DeepReplay\{}_{}_histogram.jpg'.format(species,figname))
    
    plt.figure()
    plt.plot(np.arange(0,1,0.01),np.array([np.quantile(val1,i/100) for i in range(100)]))
    plt.plot(np.arange(0,1,0.01),np.array([np.quantile(val2,i/100) for i in range(100)]))
    
    if figname: plt.savefig(r'C:\Users\George\Documents\Figures\DeepReplay\{}_{}_QuantilePlot.jpg'.format(species,figname))
    
def read_starttime(root):
    """ read datetime and convert to object from txt file """
    startfile = open(r'{}\start.txt'.format(root),"r")
    starttime = startfile.read()
    startfile.close()
    return datetime.strptime(starttime,'%d/%m/%Y %H:%M:%S')

def correct_index(df,start,correction = 0.997):
    """ Account for time warp error in video generation """
    ind = np.array((df.index - df.index[0]).total_seconds() * correction)
    ind = pd.to_datetime(ind*pd.Timedelta(1,unit = 's') + pd.to_datetime(start))
    df_ = df.copy()
    df_.index = ind
    return df_

#%% Code

"""
% of time higher then the other
average difference between values 
"""

if __name__ == '__main__':
    
    specie = {'E': 'Erpobdella','G':'Gammarus','R':'Radix'}
    time_correction = 0.997
    
    for r in [roots[0]]:
        
        Tox = int(r.split('_')[0])
        
        stem = [d for d in os.listdir(r'I:\TXM{}-PC'.format(Tox)) if r.split('_')[-1] in d]
        root = r'I:\TXM{}-PC\{}'.format(Tox,stem[0])
        starttime = read_starttime(root)
        
        #locate original and copy
        file_og = r'{}\{}.xls.zip'.format(root,stem[0])
        file_copy = r'{}\{}.replay.xls.zip'.format(root,stem[0])
        
        #read file
        df_og,df_copy = d_.read_merge([file_og]),d_.read_merge([file_copy])
        dfs_og,dfs_copy = d_.preproc(df_og),d_.preproc(df_copy)
        
        for s in specie:
            dfs_copy[s] = correct_index(dfs_copy[s], starttime, time_correction)
        
        species = 'R'
        df1,df2 = dfs_og[species],dfs_copy[species]
        df1_m,df2_m = d_.rolling_mean(df1,5),d_.rolling_mean(df2,5)
        
        #%% Read in calibration scales and add to preprocessing
        
        #%% Start with Radix
        
        t_ind1,t_ind2 = np.array((df1.index - df1.index[0]).total_seconds()),np.array((df2.index - df2.index[0]).total_seconds())
        tm_ind1,tm_ind2 = np.array((df1_m.index - df1_m.index[0]).total_seconds()),np.array((df2_m.index - df2_m.index[0]).total_seconds())
        
        fig,axe = plt.subplots(4,4,figsize = (12,20),sharex = True)
        for i in range(16):
            axe[i//4,i%4].plot(t_ind1,df1[i+1])
            axe[i//4,i%4].plot(t_ind2,df2[i+1])
        fig.tight_layout()
        
        fig,axe = plt.subplots(4,4,figsize = (12,20),sharex = True,sharey = True)
        for i in range(16):
            axe[i//4,i%4].plot(tm_ind1,df1_m[i+1])
            axe[i//4,i%4].plot(tm_ind2,df2_m[i+1])
            axe[i//4,i%4].set_ylim([0,800])
        fig.tight_layout()
        
        #plot distribution comparison
        values1,values2 = df1.values.flatten(),df2.values.flatten()
        values1,values2 = values1[values1 > 0],values2[values2 > 0]
        
        plot_distribution(values1,values2,species,figname = r)
        
        #%% Erpobdella
        species = 'E'
        df1,df2 = dfs_og[species],dfs_copy[species]
        indexing = min(df1.shape[0],df2.shape[0])
        df1,df2 = df1.iloc[:indexing],df2.iloc[:indexing]
        df1_m,df2_m = d_.rolling_mean(df1,5),d_.rolling_mean(df2,5)
        
        fig,axe = plt.subplots(4,4,figsize = (12,20),sharex = True)
        for i in range(16):
            axe[i//4,i%4].plot(np.arange(df1.shape[0]),df1[i+1])
            axe[i//4,i%4].plot(np.arange(df2.shape[0]),df2[i+1])
        fig.tight_layout()
        
        fig,axe = plt.subplots(4,4,figsize = (12,20),sharex = True,sharey = True)
        for i in range(16):
            axe[i//4,i%4].plot(np.arange(df1_m.shape[0]),df1_m[i+1])
            axe[i//4,i%4].plot(np.arange(df2_m.shape[0]),df2_m[i+1])
            axe[i//4,i%4].set_ylim([0,800])
        fig.tight_layout()
        
        #plot distribution comparison
        values1,values2 = df1.values.flatten(),df2.values.flatten()
        values1,values2 = values1[values1 > 0],values2[values2 > 0]
        
        plot_distribution(values1,values2,species,figname = r)