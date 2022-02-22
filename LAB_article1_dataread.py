# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 19:28:16 2022

Read and plot graphs for IGT on the methomyl experiments.

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