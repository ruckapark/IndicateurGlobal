# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 14:55:28 2023

csv files for individual species - already treated read from the NAS

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

def find_dopage_entry(dope_df,root):
    row = None
    for i in range(dope_df.shape[0]):
        if root.split('\\')[-1] in dope_df.iloc[i]['root']: 
            row = i
            break
    if row:
        return dope_df.iloc[row]
    else:
        print('No dopage found')
        return row

def preproc_csv(df):
    #timestamp index and integer columns
    df.index = pd.to_datetime(df.index)
    df.columns = [int(c) for c in df.columns]
    return df

def main(root):
    
    specie = {'E': 'Erpobdella','G':'Gammarus','R':'Radix'}
    dfs = {}
    rootfile_stem = root + r'\\' + root.split('\\')[-1].split('-')[0] + '_'
    
    try:
        morts = d_.read_dead(root)
    except:
        print('No mortality register')
        os._exit()
        
    for s in specie:
        species = specie[s]
        if len(morts[s]) > 11:
            print('\n Excessive mortality: ',species)
            continue
        
        #read in dataframe
        df = pd.read_csv(r'{}{}.csv.zip'.format(rootfile_stem,species),index_col = 0)
        df = preproc_csv(df)
        
        #check for unnoticed mortality
        print('Check mortality for: ',species,d_.check_dead(df,s))
        
        rewrite_csv = False
        for m in morts[s]:
            if m in df.columns: 
                df = df.drop(columns = [m])
                rewrite_csv = True
        if rewrite_csv: 
            d_.write_csv(df,s,root)         
            print('Youre in!')
            
        dfs.update({s:df})
    return dfs

#%% main code
if __name__ == '__main__':
    
    specie = {'E': 'Erpobdella','G':'Gammarus','R':'Radix'}
    dfs = {}
    root = r'I:\TXM760-PC\20210520-224501'
    rootfile_stem = root + r'\\' + root.split('\\')[-1].split('-')[0] + '_'
    
    try:
        morts = d_.read_dead(root)
    except:
        print('No mortality register')
        os._exit()
        
    for s in specie:
        species = specie[s]
        if len(morts[s]) > 11:
            print('\n Excessive mortality: ',species)
            continue
        
        #read in dataframe
        df = pd.read_csv(r'{}{}.csv.zip'.format(rootfile_stem,species),index_col = 0)
        df = preproc_csv(df)
        
        #check for unnoticed mortality
        print('Check mortality for: ',species,d_.check_dead(df,s))
        
        rewrite_csv = False
        for m in morts[s]:
            if m in df.columns: 
                df = df.drop(columns = [m])
                rewrite_csv = True
        if rewrite_csv: 
            d_.write_csv(df,s,root)         
            print('Youre in!')
            
        dfs.update({s:df})
            
    #%% Find dopage
    dope_df = dope_read_extend()
    dopage_entry = find_dopage_entry(dope_df, root)
    dopage = dopage_entry['Start']    
    
    #%%
    s = 'E'
    df = dfs[s]
    
    #mean treatment of data
    t_mins = 5
    df_mean = d_.rolling_mean(df,t_mins)
    
    d_.plot_16(df_mean,mark = [dopage_entry['Start'],dopage_entry['End']])
    
    #IGT
    quantile_distRAW = df.quantile(q = 0.10, axis = 1)**2
    quantile_dist = df_mean.quantile(q = 0.10, axis = 1)**2
    
    fig,axe = d_.single_plot(quantile_dist)
    axe.axvline(dopage,color = 'red')