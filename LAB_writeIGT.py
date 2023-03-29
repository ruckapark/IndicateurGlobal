# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 11:28:28 2023

Read datafile from zip file and write IGT file accordingly

- Test this works for single file
- Then test for joined file

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
from dope_reg import dope_read
import dataread as d_
os.chdir('..')

#%% FUNCTIONS

if __name__ == '__main__':
    
    #Could be in two directories for long entry
    root = [r'I:\TXM760-PC\20210716-082422']
    Tox = int(root[0].split('TXM')[-1][:3])
    
    specie = {'E': 'Erpobdella','G':'Gammarus','R':'Radix'}
    
    files = []
    for r in root:
        os.chdir(r) #not necessary
        file = [file for file in os.listdir() if 'xls.zip' in file]
        files.extend(file)
     
    df = d_.read_merge(files)
    dfs = d_.preproc(df)
    
    dope_df = dope_read()
    #dopage,date_range,conc,sub,molecule,etude = d_.dope_params(dope_df,Tox,df.index[0],df.index[-1]) #Cleaner would be to put here
    
    for species in [*specie]:
        
        try:
            df = dfs[species]
            dopage,date_range,conc,sub,molecule,etude = d_.dope_params(dope_df,Tox,df.index[0],df.index[-1])
        except:
            continue
        
        t_mins = 5
        df_mean = d_.rolling_mean(df,t_mins)
        
        mean_dist = df_mean.mean(axis = 1)
        quantile_dist = df_mean.quantile(q = 0.05, axis = 1)**2
        
        IGT = quantile_dist[(quantile_dist.index > dopage - pd.Timedelta(seconds = 10)) & (quantile_dist.index < dopage + pd.Timedelta(hours = 12))]
        IGT.index = ((IGT.index - IGT.index[0]).total_seconds()).astype(int)
        IGT.to_csv(r'{}\IGT_{}.csv'.format(root[-1],species),header = False)
        print('{} has {} entries'.format(specie[species],len(IGT)))