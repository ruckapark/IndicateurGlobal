# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 15:29:32 2022

First attemp at fingerprints

@author: George
"""

#%% IMPORTS classic mods
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
#import gen_pdf as pdf
import dataread as d_
os.chdir('..')

specie = {'E': 'Erpobdella','G':'Gammarus','R':'Radix'}

#%% code
if __name__ == "__main__":
    
    directory = r'D:\VP\ARTICLE1_copper\Data' #methomyl or copper
    substance = 'copper' #meth or copper
    
    #Article1 data
    os.chdir(directory)
    
    #compressed files have " _ " in file name
    files = [f for f in os.listdir() if '_' in f]
    #files = [files[0]]
    dope_df = dope_read('{}_reg'.format(substance))
    
    
    #treat each data file, add to dictionary
    data = {}
    dopages = {}
    for file in files:
        
        #Toxname
        Tox = int(file[:3])
        
        #extract only gammarus df
        df = d_.read_merge([file])
        dfs = d_.preproc(df)
        df = dfs['G']
        deadcol = d_.remove_dead(df,'G')
        if len(deadcol) > 14: continue
        df = df.drop(columns = deadcol)
        
        #meaned df
        t_mins = 5
        df_mean = d_.rolling_mean(df,t_mins)
        
        #find dopage time
        dopage,date_range,conc,sub,molecule,etude_ = d_.dope_params(dope_df,Tox,df.index[0],df.index[-1])
        
        data.update({file:[df,df_mean]})
        dopages.update({file:[dopage,date_range,conc]})
        
    #%% analysis
    for study in data:
        
        #data
        [df,df_mean] = data[study]
        [dopage,date_range,conc] = dopages[study]
        
        mean_dist = df_mean.mean(axis = 1)
        quantile_dist = df_mean.quantile(q = 0.05, axis = 1)**2
        
        fig,axe = d_.single_plot(mean_dist,title = 'Mean : {}'.format(conc))
        d_.dataplot_mark_dopage(axe,date_range)
        
        fig,axe = d_.single_plot(quantile_dist,title = 'IGT : {}'.format(conc))
        d_.dataplot_mark_dopage(axe,date_range)
        
        #define first changepoint
        changepoint = dopage
        
        #define first max point over 0.99
        uplimit = quantile_dist.quantile(0.99)
        if uplimit < 500:
            time_up = 0
            time_down = 0
            endtime_down = 0
            upgradient = 0
            downgradient = 0
            timelag = 0
            timepeak = 0
            timehighpeak = 0
        else:
            downlimit = 50
            quantup = quantile_dist[quantile_dist > uplimit]
            quantdown = quantile_dist[quantile_dist < downlimit]
            time_up = quantup.index[0]
        
            #define descent point
            time_down = quantup.index[-1]
            #define last descent point
            quantdown = quantdown[quantdown.index > time_down]
            try:
                endtime_down = quantdown.index[0]
            except:
                endtime_down = quantile_dist.index[-1]
            
            #define gradient up
            upgradient = uplimit/(time_up - changepoint).total_seconds()
            
            #define gradient down
            downgradient = uplimit/(endtime_down - time_down).total_seconds()
            
            #define timelag to reaction - use changepoint from bioessais
            timelag = (changepoint - dopage).total_seconds()
            timepeak = (endtime_down - changepoint).total_seconds()
            timehighpeak = (time_down - time_up).total_seconds()
            
            axe.axvline(time_up, color = 'red')
            axe.axvline(time_down, color = 'red')
            axe.axvline(endtime_down, color = 'red')
        
        #define distribution evolution