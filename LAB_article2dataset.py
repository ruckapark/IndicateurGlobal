# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 18:44:50 2023

Generate dataset for ToxPrintsdatasets

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

"""
Zinc selection:
    116 no gammarus data
    121 good: short peak to 12000
    122 nothing on IGT mean ok
    125 nothing on IGT mean ok
    130 nothing on IGT mean ok
    131 good: short peak to 1400 - not great on mean
    144 good: sustained peak to 3000
    145 no data
    155 good: sustained peak to 8000
    156 no data
    159 good: short peak to 20000
"""

if __name__ == '__main__':
    
    #213,224 not cooperating
    studies = {'Cuivre':[176,127,133,158],
             'Zinc':[121,144,155,159],
             'Lindane':[211,212],
             'alphaEndosulfan':[251,252,260,261],
             'betaEndosulfan':[253,254,262,263]}
    
    specie = {'E': 'Erpobdella','G':'Gammarus','R':'Radix'}
    
    dope_df = dope_read_extend()
    
    #add methomyl one in I drive from replays
    substances = ['Cuivre','Zinc','Lindane','alphaEndosulfan','betaEndosulfan']
    concentrations = dict(zip(substances[:2],['100ug','324ug']))
    sub_dope_df = dope_df[dope_df['Substance'].isin(substances)]
    
    dataset = r'D:\VP\ARTICLE2\Data'
    for substance in studies:
        for x,i in enumerate(studies[substance]):
            Tox = dope_df.iloc[i]['TxM']
            root = [r'I:\TXM{}-PC\{}'.format(Tox,r) for r in dope_df.iloc[i]['root']]
            
            files = []
            for r in root:
                file = [r'{}\{}'.format(r,file) for file in os.listdir(r) if 'xls.zip' in file]
                files.extend(file)
                
            
            df = d_.read_merge(files)
            dfs = d_.preproc(df)
        
            for species in ['G']:
                
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
                meansRAW = mean_distRAW[(mean_distRAW.index >= 0) & (mean_distRAW.index < 12*3600)]
                IGT_RAW = quantile_distRAW[(quantile_distRAW.index >= 0) & (quantile_distRAW.index < 12*3600)]
                
                means = mean_dist[(mean_dist.index >= 0) & (mean_dist.index < 12*3600)]
                IGT = quantile_dist[(quantile_dist.index >= 0) & (quantile_dist.index < 12*3600)]
                
                # plt.figure()
                # plt.plot(IGT)
                # plt.title('IGT {}'.format(i))
                
                # plt.figure()
                # plt.plot(means)
                # plt.title('Mean {}'.format(i))
                
                means.to_csv(r'{}\{}_means{}{}.csv'.format(dataset,substance[0].upper(),species,x),header = False)
                IGT.to_csv(r'{}\{}_IGT{}{}.csv'.format(dataset,substance[0].upper(),species,x),header = False)