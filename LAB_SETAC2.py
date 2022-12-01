# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 13:38:54 2022

Code for SETAC 2 Figures

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

#%% FUNCTIONS - plotting functions could also be put into another module  

def scale_IGT(IGT):
    m = np.max(IGT)
    return IGT/m

def zero_x(x,dopage):
    """ for timeseries; zero t at t = dopage """
    x.index = ((x.index - dopage).total_seconds() / 3600)
    return x

if __name__ == '__main__':
    plt.close('all')
    
    #H40 762, 2A1 768 (1mg/L)
    Tox,etude = 768,54
    specie = {'R':'Radix','E': 'Erpobdella','G':'Gammarus'}
    spec_colors = {'E':'#1d991d','G':'#e0aa07','R':'#077be0'}
    
    os.chdir(r'D:\VP\Viewpoint_data\TxM{}-PC\{}'.format(Tox,d_.study_no(etude)))
    files = [file for file in os.listdir() if os.path.isfile(file)]
     
    df = d_.read_merge(files)
    dfs = d_.preproc(df)
    
    dope_df = dope_read()
    
    #%%
    
    IGT = {}
    MEANS = {}
    
    fig = plt.figure(figsize = (10,8))
    axe = fig.add_axes([0.1,0.1,0.8,0.8])
    axe.axvline(color = 'r')
    axe.set_title('Avoidance to Surfactant 2',fontsize = 18)
    axe.axes.yaxis.set_ticklabels([])
    axe.set_ylabel('Standardised bioactivity signal (mm/s)',fontsize = 16)
    axe.set_xlabel('Exposure time (hours)', fontsize = 16)
    
    # could also seperate out over time
    for species in [*specie]:
        df = dfs[species]
        df = df.drop(columns = d_.remove_dead(df,species))
        
        dopage,date_range,conc,sub,molecule,etude = d_.dope_params(dope_df,Tox,df.index[0],df.index[-1])
        
        # moving means (look at block and mobile mean combination)
        t_mins = 10
        df_mean = d_.rolling_mean(df,t_mins)
        
        #plot all the means across cells
        mean_dist = df_mean.mean(axis = 1)
        quantile_dist = df_mean.quantile(q = 0.05, axis = 1)**2
        
        #smaller plot
        IGT.update({species : quantile_dist[(quantile_dist.index > dopage - pd.Timedelta(minutes = 15)) & (quantile_dist.index < dopage + pd.Timedelta(minutes = 120))]})
        MEANS.update({species : mean_dist[(mean_dist.index > dopage - pd.Timedelta(hours = 1)) & (mean_dist.index < dopage + pd.Timedelta(hours = 2))]})
        
        igt = zero_x(scale_IGT(IGT[species]),dopage)
        axe.plot(igt,color = spec_colors[species])
        axe.fill_between(igt.index,igt,color = spec_colors[species],alpha = 0.2)