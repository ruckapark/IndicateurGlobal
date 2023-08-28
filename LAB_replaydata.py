# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 11:20:36 2023

Use the replay filter functions from LAB_replay_//
-timelag for Gammarus
-ERPO for Erpobdella
-RADIX for Radix

Check results are coherent

@author: George
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from sklearn import preprocessing
from datetime import timedelta,datetime
from scipy import signal

#%% IMPORT personal mods
os.chdir('MODS')
from data_merge import merge_dfs
from dope_reg import dope_read_extend
import dataread as d_
os.chdir('..')

#%% IMPORT replay files
from LAB_replay_ERPO import filter_erpo
#from LAB_replay_timelag import filter_gammarus
from LAB_replay_RADIX import filter_radix

#%% Relevant directories
roots = ['765_20211022',
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

if __name__ == "__main__":
    
    specie = {'E': 'Erpobdella','G':'Gammarus','R':'Radix'}
    time_correction = 0.997
    values_old,values_new = np.array([]),np.array([])
    
    #can make loop for all roots after
    r = roots[0]
    Tox = int(r.split('_')[0])
    stem = [d for d in os.listdir(r'I:\TXM{}-PC'.format(Tox)) if r.split('_')[-1] in d]
    root = r'I:\TXM{}-PC\{}'.format(Tox,stem[0])
    starttime = d_.read_starttime(root)
    
    #read old and new xls file
    file_og = r'{}\{}.xls.zip'.format(root,stem[0])
    file_copy = r'{}\{}.replay.xls.zip'.format(root,stem[0])
    
    #read file, wrangle and calibrate
    df_og,df_copy = d_.read_merge([file_og]),d_.read_merge([file_copy])
    dfs_og,dfs_copy = d_.preproc(df_og),d_.preproc(df_copy)
    dfs_og,dfs_copy = d_.calibrate(dfs_og,Tox,starttime),d_.calibrate(dfs_copy,Tox,starttime)
    
    #correct time index including time warp, original if necessary
    if datetime.strptime(r.split('_')[-1],'%Y%m%d') == starttime.replace(hour=0, minute=0, second=0):
        reset_original = False
    else:
        reset_original = True
    for s in specie:
        dfs_copy[s] = d_.correct_index(dfs_copy[s], starttime, time_correction)
        if reset_original: d_.correct_index(dfs_og[s], starttime, correction = 1)
        
    #read dead values
    morts = d_.read_dead(root)
    dfs_og,dfs_copy = d_.remove_dead_known(dfs_og,morts),d_.remove_dead_known(dfs_copy,morts)
    
    #%% Gammarus
    
    #%% Erpobdella
    species = 'E'
    df1,df2 = dfs_og[species],dfs_copy[species]
    
    #read in quantization and check for count of mid bursts
    df_quant_mid = d_.read_quant([file_og])
    df_q = d_.preproc(df_quant_mid,quant = True)[species]
    
    #get seconds time indexes and plot original graph with quant figure
    t_ind1,t_ind2 = np.array((df1.index - df1.index[0]).total_seconds()),np.array((df2.index - df2.index[0]).total_seconds())
    fig,axe = plt.subplots(4,4,figsize = (12,20),sharex = True)
    axe_q = np.empty(axe.shape,dtype = object)
    for i in range(16):
        axe_q[i//4,i%4] = axe[i//4,i%4].twinx()
        if i+1 not in df1.columns: continue
        axe[i//4,i%4].plot(t_ind1,df1[i+1])
        axe[i//4,i%4].plot(t_ind2,df2[i+1])
        axe_q[i//4,i%4].plot(t_ind1,df_q[i+1],color = 'r',alpha = 0.3)
    fig.tight_layout()
    
    df_r = filter_erpo(df2,df1,df_q)
    
    #plot amended time series
    fig,axe = plt.subplots(4,4,figsize = (12,20),sharex = True)
    for i in range(16):
        if i+1 not in df1.columns: continue
        axe[i//4,i%4].plot(t_ind1,df1[i+1])
        axe[i//4,i%4].plot(t_ind2,df_r[i+1],color = 'red',alpha = 0.75)
    fig.tight_layout()
    
    #%% Radix
    species = 'R'
    df1,df2 = dfs_og[species],dfs_copy[species]
    
    #get seconds time indexes and plot original graph with quant figure
    t_ind1,t_ind2 = np.array((df1.index - df1.index[0]).total_seconds()),np.array((df2.index - df2.index[0]).total_seconds())
    fig,axe = plt.subplots(4,4,figsize = (12,20),sharex = True)
    axe_q = np.empty(axe.shape,dtype = object)
    for i in range(16):
        axe_q[i//4,i%4] = axe[i//4,i%4].twinx()
        if i+1 not in df1.columns: continue
        axe[i//4,i%4].plot(t_ind1,df1[i+1])
        axe[i//4,i%4].plot(t_ind2,df2[i+1])
    fig.tight_layout()
    
    df_r = filter_radix(df2,df1)
    
    #plot amended time series
    fig,axe = plt.subplots(4,4,figsize = (12,20),sharex = True)
    for i in range(16):
        if i+1 not in df1.columns: continue
        axe[i//4,i%4].plot(t_ind1,df1[i+1])
        axe[i//4,i%4].plot(t_ind2,df_r[i+1],color = 'red',alpha = 0.75)
    fig.tight_layout()