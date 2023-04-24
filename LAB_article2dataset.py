# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 18:44:50 2023

Generate dataset for ToxPrintsdatasets

@author: George
"""

"""
Dataset selection SETAC

his should be the priority list of files to exclude

Copper files for erpo and radix
Zinc files for erpo and radix
Methomyl files for erpo and radix ??

Study numbers - Cu (70,107,109,111,112,118,123,127,132,133,142,143,158,168,176)
Study numbers - Zn (116,121,122,125,130,131,144,145,155,156,159)
Study numbers - Me (136,137,147,148,151,152,153,154,165,166,209,210)

Studies:

Cu
-70  E: ok,ok G: ok,ok R: ok,ok
-107 E: ok,ok G: ok,ok R: ok,ok
-109 E: ok,ok G: ok,ok R: ok,ok
-111 E: no,no G: --,-- R: ok,ok
-112 E: no,-- G: ok,ok R: ok,ok
-118 E: ok,ok G: ok,ok R: --,--
-123 E: no,no G: ok,ok R: --,--
-127 E: ok,ok G: ok,ok R: no,no
-132 E: --,ok G: --,ok R: ok,ok
-133 E: ok,ok G: ok,ok R: --,--
-142 E: --,ok G: ok,ok R: ok,ok
-143 E: ok,ok G: ok,ok R: --,ok
-158 E: ok,ok G: ok,ok R: ok,--
-168 E: no,ok G: ok,ok R: ok,--
-176 E: no,ok G: ok,ok R: no,ok

Selected cases:
70,107,109,142,143
best Erpobdella - 70,107,118,127,133,143,158,
best Gammarus - 70,107,109,112,123,127,133,142,143,158,168,176
best Radix - 70,107,109,111,112,132,142,168
alternative radix type - 118,123,133,158

Zn
-na 116 E: -- G: R: 
-121 E: --,-- G: ok,ok R: --,--
-122 E: --,-- G: --,ok R: --,--
-125 E: --,ok G: --,ok R: --,--
-130 E: --,no G: --,ok R: ok,ok
-131 E: ok,ok G: ok,ok R: --,--
-144 E: no,no G: ok,ok R: no,no
-na 145 E: G: R:
-155 E: ok,ok G: ok,ok R: --,--
-na 156 E: G: R:
-159 E: no,no G: ok,ok R: --,--

Selected cases:
121,131,144,159
best Erpobdella - not 144 all others ok
best Gammarus - good means with some missing IGT
best Radix - all fine
Radix and erpo seem to have no reaction at this dose

Me
-na 136 E: G: R:
-na 137 E: G: R:
-na 147 E: G: R:
-na 148 E: G: R:
-na 151 E: G: R:
-na 152 E: G: R:
-na 153 E: G: R:
-na 154 E: G: R:
all used - 209 radix has false detection
-165 E: G: R:
-166 E: G: R:
-209 E: G: R:
-210 E: G: R:

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
    
    studies_SETAC = {
        'Cu':[70,107,109,111,112,118,123,127,132,133,142,143,158,168,176],
        'Zi':[121,122,125,130,131,155,159],
        'Me':[165,166,209,210]
        }
    
    specie = {'E': 'Erpobdella','G':'Gammarus','R':'Radix'}
    
    dope_df = dope_read_extend()
    
    #add methomyl one in I drive from replays
    substances = ['Cuivre','Zinc','Lindane','alphaEndosulfan','betaEndosulfan']
    concentrations = dict(zip(substances[:2],['100ug','324ug']))
    sub_dope_df = dope_df[dope_df['Substance'].isin(substances)]
    
    dataset = r'D:\VP\ARTICLE2\Data'
    datasetSETAC = r'D:\VP\ARTICLE2\SETAC'
    
    studies = studies_SETAC
    
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
        
            for species in [*specie]:
                
                dataset = r'{}\{}'.format(datasetSETAC,specie[species])
                
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