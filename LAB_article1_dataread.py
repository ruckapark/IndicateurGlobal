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

def find_reaction(df,dopage):
    reaction_period = df[df.index > dopage + pd.Timedelta(hours = 1)]
    try:
        return reaction_period[reaction_period > 100].index[0]
    except:
        return None

#%% code
if __name__ == "__main__":
    
    directory = r'D:\VP\ARTICLE1_methomyl\Data' #methomyl or copper
    substance = 'meth' #meth or copper
    
    #Article1 data
    os.chdir(directory)
    
    #compressed files have " _ " in file name
    files = [f for f in os.listdir() if '_' in f]
    # files = ['760_Methomyl4.xls','760_Methomyl5.xls','761_Methomyl4.xls',
    #          '761_Methomyl5.xls','761_Methomyl6.xls','762_Methomyl4.xls',
    #          '768_Methomyl2.xls','769_Methomyl2.xls']
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
        #if conc != '67ug': continue
        
        data.update({file:[df,df_mean]})
        dopages.update({file:[dopage,date_range,conc]})
        
    #%% analysis
    for study in data:
        
        #data
        [df,df_mean] = data[study]
        [dopage,date_range,conc] = dopages[study]
        
        if conc == '125ug':
            mean_dist = df_mean.mean(axis = 1)
            quantile_dist = df_mean.quantile(q = 0.05, axis = 1)**2
            
            fig,axe = d_.single_plot(mean_dist,title = 'Mean : {} - {}'.format(conc,study))
            d_.dataplot_mark_dopage(axe,date_range)
            
            fig,axe = d_.single_plot(quantile_dist,title = 'IGT : {} - {}'.format(conc,study))
            d_.dataplot_mark_dopage(axe,date_range)
            
            #find reaction point using 100
            start_reaction = find_reaction(quantile_dist,dopage)
            if start_reaction: axe.axvline(start_reaction,color = 'orange')
            
            #find reaction point using % of max non-zero value
            