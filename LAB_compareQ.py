# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 10:10:12 2021



@author: Admin
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

#%% FUNCTIONS - plotting functions could also be put into another module         

if __name__ == '__main__':
                
    Tox,species,etude = 768,'G',64
    specie = {'E': 'Erpobdella','G':'Gammarus','R':'Radix'}
    
    os.chdir(r'D:\VP\Viewpoint_data\TxM{}-PC\{}'.format(Tox,d_.study_no(etude)))
    files = [file for file in os.listdir() if os.path.isfile(file)]
     
    df = d_.read_merge(files)
    dfs = d_.preproc(df)
    
    fig = plt.figure(figsize = (12,8))
    axe = fig.add_axes([0.1,0.1,0.8,0.8])
    
    dope_df = dope_read()
        
    # could also seperate out over time
    df = dfs[species]
    df = df.drop(columns = d_.remove_dead(df,species))
    
    dopage,date_range,conc,sub,molecule,etude = d_.dope_params(dope_df,Tox,df.index[0],df.index[-1])
    
    # moving means
    t_mins = 15
    df_mean = d_.rolling_mean(df,t_mins)
    quantile_dist1 = df_mean.quantile(q = 0.05, axis = 1)**2
    quantile_dist2 = df_mean.quantile(q = 0.1, axis = 1)**2
    quantile_dist3 = df_mean.quantile(q = 0.25, axis = 1)**2
    
    title_ = 'ToxIndex comparison'
    
    #smaller plot
    IGT1 = quantile_dist1[(quantile_dist1.index > dopage - pd.Timedelta(minutes = 60)) & (quantile_dist1.index < dopage + pd.Timedelta(minutes = 120))]
    IGT2 = quantile_dist2[(quantile_dist2.index > dopage - pd.Timedelta(minutes = 60)) & (quantile_dist2.index < dopage + pd.Timedelta(minutes = 120))]
    IGT3 = quantile_dist3[(quantile_dist3.index > dopage - pd.Timedelta(minutes = 60)) & (quantile_dist3.index < dopage + pd.Timedelta(minutes = 120))]

    axe.plot(IGT1,label = '5%')
    axe.plot(IGT2,label = '10%')
    axe.plot(IGT3,label = '25%')
    axe.axvline(dopage,color = 'r')
    
    axe.legend()
    axe.set_title(title_)