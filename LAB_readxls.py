# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 16:47:07 2023

Single study read IGT

@author: George
"""

#%% IMPORTS

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from datetime import timedelta

#%% IMPORT personal mods
os.chdir('MODS')
from data_merge import merge_dfs
from dope_reg import dope_read_extend
import dataread as d_
os.chdir('..')

#%% FUNCTIONS

def IGT(df,dopage = None):
    IGT = df.quantile(q = 0.05, axis = 1)**2
    if dopage:
        if type(df.index) == pd.core.indexes.datetimes.DatetimeIndex:
            time = np.array((df.index - df.index[0]).total_seconds())/60
            plt.figure()
            plt.plot(time,np.array(IGT),label = 'IGT')
            plt.axvline((dopage - df.index[0]).total_seconds()/60,label = dopage.strftime('%d/%m/%Y %H:%M:%S'),color = 'r', linestyle = '--')
            plt.legend()
        elif type(df.index) == pd.core.indexes.base.Index:
            if type(dopage) == pd._libs.tslibs.timestamps.Timestamp:
                print('Dopage should be numeric in seconds')
            else:
                plt.figure()
                plt.plot(IGT.index,np.array(IGT),label = 'IGT')
                plt.axvline(dopage/60,label = dopage.strftime('{} minutes'.format(dopage/60)),color = 'r',linstyle = '--')
                plt.legend()
    return IGT

def mean():
    mean = df.mean(axis = 1)
    if dopage:
        if type(df.index) == pd.core.indexes.datetimes.DatetimeIndex:
            time = np.array((df.index - df.index[0]).total_seconds())/60
            plt.figure()
            plt.plot(time,np.array(mean),label = 'mean movement')
            plt.axvline((dopage - df.index[0]).seconds/60,label = dopage.strftime('%d/%m/%Y %H:%M:%S'))
            plt.legend()
        elif type(df.index) == pd.core.indexes.base.Index:
            if type(dopage) == pd._libs.tslibs.timestamps.Timestamp:
                print('Dopage should be numeric in seconds')
            else:
                plt.figure()
                plt.plot(mean.index,np.array(mean),label = 'mean')
                plt.axvline(dopage/60,label = dopage.strftime('{} minutes'.format(dopage/60)))
                plt.legend()
    return IGT

#%% main

if __name__ == '__main__':
    
    specie = {'E': 'Erpobdella','G':'Gammarus','R':'Radix'}
    
    dope_df = dope_read_extend()
    
    i = int(input('Study number?'))
    
    Tox = dope_df.iloc[i]['TxM']
    root = [r'I:\TXM{}-PC\{}'.format(Tox,r) for r in dope_df.iloc[i]['root']]

    files = []
    for r in root:
        file = [r'{}\{}'.format(r,file) for file in os.listdir(r) if 'xls.zip' in file]
        files.extend(file)
        
    
    df = d_.read_merge(files)
    dfs = d_.preproc(df)

    for species in [*specie]:
        
        df = dfs[species]
        dopage,date_range,conc,sub,molecule,etude = d_.dope_params(dope_df,Tox,df.index[0],df.index[-1])
        
        t_mins = 5
        df_mean = d_.rolling_mean(df,t_mins)
        
        #add non meaned
        
        mean_distRAW = df.mean(axis = 1)
        mean_dist = df_mean.mean(axis = 1)
        
        #zero time reference for dopage
        zeroRAW = abs((dopage - mean_distRAW.index).total_seconds()).argmin()
        zero = abs((dopage - mean_dist.index).total_seconds()).argmin()
        
        mean_distRAW.index = ((mean_distRAW.index - mean_distRAW.index[zeroRAW]).total_seconds()).astype(int)
        mean_dist.index = ((mean_dist.index - mean_dist.index[zero]).total_seconds()).astype(int)
        
        quantile_distRAW = df.quantile(q = 0.05, axis = 1)**2
        quantile_dist = df_mean.quantile(q = 0.05, axis = 1)**2
        quantile_distRAW.index = mean_distRAW.index
        quantile_dist.index = mean_dist.index
        
        #claculate IGT
        IGT_RAW = quantile_distRAW[(quantile_distRAW.index >= 0) & (quantile_distRAW.index < 12*3600)]
        IGT = quantile_dist[(quantile_dist.index >= 0) & (quantile_dist.index < 12*3600)]