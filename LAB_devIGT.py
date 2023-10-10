# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 13:44:38 2023

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

methomyls = [
    r'I:\TXM760-PC\20210520-224501',
    r'I:\TXM760-PC\20210625-093621',
    r'I:\TXM761-PC\20210520-224549',
    r'I:\TXM761-PC\20210625-093641'
    ]

dics = [
    r'I:\TXM765-PC\20211022-095148',
    r'I:\TXM767-PC\20211022-100308',
    r'I:\TXM765-PC\20211029-075240'      
    ]

#%% main code
if __name__ == '__main__':
    
    plt.close('all')
    
    specie = {'E': 'Erpobdella','G':'Gammarus','R':'Radix'}
    dfs = {}
    root = dics[2]
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
    
    #%% IGT is the IGT of the filtered series (is this the best option?)
    s = 'E'
    df = dfs[s]
    
    #mean treatment of data
    t_mins = 5
    df_mean = d_.rolling_mean(df,t_mins)
    
    d_.plot_16(df_mean,mark = [dopage_entry['Start'],dopage_entry['End']])
    
    #lower quantile
    quantile_distRAW = df.quantile(q = 0.10, axis = 1)**2
    quantile_dist = df_mean.quantile(q = 0.10, axis = 1)**2    
    
    #Calcul du signal pour avec 95% des replicats les moins actifs (quantile 95%)
    #only take values under 3 (nautral log 30)
    
    low_cutoff = 30
    maxvalue = 80
    scale_factor = maxvalue/low_cutoff
    
    quantile_low = df_mean.quantile(q = 0.90, axis = 1)
    IGT_low = quantile_low - low_cutoff
    IGT_low[IGT_low > 0] = 0.0
    IGT_low = -((IGT_low) * scale_factor)**2
    
    # low_cutoff = 30 ** (1/np.exp(1))
    # scale_factor = 10
    # quantile_low = df_mean.quantile(q = 0.90, axis = 1)
    # logQtl95 = np.log(quantile_low + 1)
    # IGT_low_activity = logQtl95 - low_cutoff
    # IGT_low_activity[IGT_low_activity > 0] = 0.0
    # IGT_low_activity = np.abs(IGT_low_activity) * scale_factor
    
    #visualise IGT
    fig,axe = d_.single_plot(quantile_dist)
    axe.plot(IGT_low.index,IGT_low)
    axe.axvline(dopage,color = 'red')
    
    #12 dichlor for radix dev.
    s = 'R'
    df = dfs[s]
    
    #mean treatment of data
    t_mins = 5
    df_mean = d_.rolling_mean(df,t_mins)
    
    d_.plot_16(df_mean,mark = [dopage_entry['Start'],dopage_entry['End']])
    
    #lower quantile
    quantile_distRAW = df.quantile(q = 0.10, axis = 1)**2
    quantile_dist = df_mean.quantile(q = 0.10, axis = 1)**2
    
    low_cutoff = 4
    maxvalue = 4
    scale_factor = maxvalue/low_cutoff
    
    quantile_low = df_mean.quantile(q = 0.90, axis = 1)
    IGT_low = quantile_low - low_cutoff
    IGT_low[IGT_low > 0] = 0.0
    IGT_low = -((IGT_low) * scale_factor)**2
    
    fig,axe = d_.single_plot(quantile_dist)
    axe.plot(IGT_low.index,IGT_low)
    axe.axvline(dopage,color = 'red')