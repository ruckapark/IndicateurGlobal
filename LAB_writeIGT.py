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
from dope_reg import dope_read_extend
import dataread as d_
os.chdir('..')

#%% FUNCTIONS

if __name__ == '__main__':
    
    specie = {'E': 'Erpobdella','G':'Gammarus','R':'Radix'}
    
    dope_df = dope_read_extend()
    #dopage,date_range,conc,sub,molecule,etude = d_.dope_params(dope_df,Tox,df.index[0],df.index[-1]) #Cleaner would be to put here
    
    for i in range(dope_df.shape[0]):
        Tox = dope_df.iloc[i]['TxM']
        root = [r'I:\TXM{}-PC\{}'.format(Tox,r) for r in dope_df.iloc[i]['root']]
        
        if 'IGT_G.csv' in os.listdir(root[-1]): continue
    
        files = []
        for r in root:
            os.chdir(r) #not necessary
            file = [file for file in os.listdir() if 'xls.zip' in file]
            files.extend(file)
     
        df = d_.read_merge(files)
        dfs = d_.preproc(df)
    
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