# -*- coding: utf-8 -*-
"""
Created on Tue May  2 18:14:15 2023

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
    
    subs = ['Acide Acrylique','L1000','H40','MHPC724','P520','A736','2A1','Soja']
    
    sub1 = dope_df[dope_df['Substance'] == 'Acide Acrylique']
    sub2 = dope_df[dope_df['Substance'] == 'L1000']
    sub3 = dope_df[dope_df['Substance'] == 'H40']
    sub4 = dope_df[dope_df['Substance'] == 'MHPC724']
    sub5 = dope_df[dope_df['Substance'] == 'P520']
    sub6 = dope_df[dope_df['Substance'] == 'A736']
    sub7 = dope_df[dope_df['Substance'] == '2A1']
    sub8 = dope_df[dope_df['Substance'] == 'Soja']

    file_dict = {
        subs[0]:[[sub1.iloc[i]['TxM'],sub1.iloc[i]['root']] for i in range(sub1.shape[0])],
        subs[1]:[[sub2.iloc[i]['TxM'],sub1.iloc[i]['root']] for i in range(sub1.shape[0])],
        subs[2]:[[sub3.iloc[i]['TxM'],sub1.iloc[i]['root']] for i in range(sub1.shape[0])],
        subs[3]:[[sub4.iloc[i]['TxM'],sub1.iloc[i]['root']] for i in range(sub1.shape[0])],
        subs[4]:[[sub5.iloc[i]['TxM'],sub1.iloc[i]['root']] for i in range(sub1.shape[0])],
        subs[5]:[[sub6.iloc[i]['TxM'],sub1.iloc[i]['root']] for i in range(sub1.shape[0])],
        subs[6]:[[sub7.iloc[i]['TxM'],sub1.iloc[i]['root']] for i in range(sub1.shape[0])],
        subs[7]:[[sub8.iloc[i]['TxM'],sub1.iloc[i]['root']] for i in range(sub1.shape[0])]
        }
    
    index_dict = {
        subs[0]:[375,385,418,505,513],
        subs[1]:[376,386,419,506,514],
        subs[2]:[377,378,387,389,420,507,515],
        subs[3]:[379,417,421,508,516],
        subs[4]:[382,388,423,510,517],
        subs[5]:[380,381,390,422,509,518],
        subs[6]:[383,391,424,512,519],
        subs[7]:[384,392,425,511,520]
        }
    
    for substance in subs:
    
        for i in index_dict[substance]:
        
            Tox = dope_df.iloc[i]['TxM']
            root = [r'I:\TXM{}-PC\{}'.format(Tox,r) for r in dope_df.iloc[i]['root']]
        
            files = []
            for r in root:
                file = [r'{}\{}'.format(r,file) for file in os.listdir(r) if 'xls.zip' in file]
                files.extend(file)
                
            try:
                df = d_.read_merge(files)
            except:
                continue
            
            dfs = d_.preproc(df)
            
            dopage = dope_df.iloc[i]['End']
            conc = dope_df.iloc[i]['Concentration']
            molecule = dope_df.iloc[i]['Molecule']
            
            fig,ax = plt.subplots(3,1,figsize = (10,15))
            
            for j,species in enumerate([*specie]):
                
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
                
                #visualise
                fig.suptitle('{} Study:{}'.format(substance,i))
                ax[j].plot(IGT)
                ax[j].set_title(specie[species])