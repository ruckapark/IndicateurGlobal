# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 11:29:48 2023

@author: George
"""

copper_entries = ['763_Copper5.xls',
                  '764_Copper4.xls',
                  '765_Copper5.xls',
                  '767_Copper7.xls',
                  '767_Copper9.xls']

meth_entries = ['760_Methomyl4.xls',
                '761_Methomyl1.xls',
                '761_Methomyl3.xls',
                '761_Methomyl5.xls',
                '769_Methomyl1.xls']

def index_hours(index,dopage):
    #return hours as float 0 = dopage
    hours = (index - dopage).total_seconds()/3600
    return hours

#%% IMPORTS classic mods
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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

def find_reaction(df,dopage = 0):
    if dopage:
        delta = pd.Timedelta(hours = 1)
    else:
        delta = 1
    reaction_period = df[df.index > dopage + delta]
    try:
        return reaction_period[reaction_period > 100].index[0]
    except:
        return None
    
def block_mean(df,timestep = 2,unit = 'm'):
    
    """
    Coded for timestep block mean and not integer
    """
    
    df = df.copy()
    df['t'] = (60*(df.index - df.index[0]))//timestep
    df_m = df.groupby('t').mean()
    df_m.index = df.index[0] + timestep/60
    return df_m

#%% code
if __name__ == "__main__":
    
    #file to compress and dope reg
    file = r'D:\VP\ARTICLE1_copper\Data\765_copper5.xls'
    sub = 'copper'
    dope_df = dope_read('{}_reg'.format(sub))
    
        
    #Toxname
    Tox = int(file.split('\\')[-1][:3])
    
    #extract only gammarus df
    df = d_.read_merge([file])
    dfs = d_.preproc(df)
    df = dfs['G']
    deadcol = d_.remove_dead(df,'G')
    df = df.drop(columns = deadcol)
    
    #meaned df
    t_mins = 2
    #df_mean = d_.rolling_mean(df,t_mins)
    df_mean = d_.block_mean(df,t_mins)
    #df_mean = d_.rolling_mean(df_mean,5)
    
    #find dopage time
    dopage,date_range,conc,subs,molecule,etude_ = d_.dope_params(dope_df,Tox,df.index[0],df.index[-1])
    
    #data.update({file:[df.set_index(index_hours(df.index, dopage)),df_mean.set_index(index_hours(df_mean.index, dopage))]})
    #dopages.update({file:[dopage,date_range,conc]})
        
    #%% analysis
    sns.set_style("white")
    
    palet = sns.color_palette()[:5]
    
    mean_dist = df_mean.mean(axis = 1)
    quantile_dist = df.quantile(q = 0.05, axis = 1)**2
    
    plt.figure()
    plt.plot(quantile_dist)
    plt.axvline(dopage,color = 'red')