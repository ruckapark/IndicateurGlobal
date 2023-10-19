# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 13:44:38 2023

Shows the occurance of dips in high activity for both radix and methomyl

This is presented in two figures from methomyl and 1,2dichlor

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
    #r'I:\TXM767-PC\20211022-100308',
    r'I:\TXM765-PC\20211029-075240'      
    ]

#%% main code
if __name__ == '__main__':
    
    specie = {'E': 'Erpobdella','G':'Gammarus','R':'Radix'}
    dfs = {}
    roots = methomyls
    
    dope_df = dope_read_extend()
    
    #IGT values
    optimum = 0.129167
    optimum2 = 0.1909
    
    #plot subplots
    fig,axes = plt.subplots(2,len(roots),figsize = (6*len(roots),13))
    axes[0,0].set_ylabel('Erpobdella')
    axes[1,0].set_ylabel('Radix')
    fig.suptitle('High quantile observations - Hypoactivity')
    
    for i,root in enumerate(roots):
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
        dopage_entry = find_dopage_entry(dope_df, root)
        dopage = dopage_entry['Start']    
        
        #%% IGT is the IGT of the filtered series (is this the best option?)
        
        s = 'E'
        df = dfs[s]
        
        #mean treatment of data
        t_mins = 5
        df_mean = d_.rolling_mean(df,t_mins)
        
        #d_.plot_16(df_mean,mark = [dopage_entry['Start'],dopage_entry['End']])
        
        #lower quantile
        quantile_distRAW = df.quantile(q = optimum, axis = 1)
        quantile_dist = d_.rolling_mean(quantile_distRAW,t_mins)
        
        #quantile_dist = df_mean.quantile(q = 0.129167, axis = 1)**2    #overestimate of quantile should use mean of raw values
        
        #Calcul du signal pour avec 95% des replicats les moins actifs (quantile 95%)
        #only take values under 3 (nautral log 30)
        
        low_cutoff = 20
        maxvalue = 70
        scale_factor = maxvalue/low_cutoff
        
        quantile_low_1 = df_mean.quantile(q = 1 - optimum, axis = 1)  #overestimate of quantile should use mean of raw values
        quantile_low_2 = d_.rolling_mean(df.quantile(q = 1 - optimum, axis = 1),t_mins)
        
        """
        IGT_low = quantile_low - low_cutoff
        IGT_low[IGT_low > 0] = 0.0
        IGT_low = -((IGT_low) * scale_factor)**2
        """
        
        
        #visualise IGT
        axes[0,i].plot(quantile_dist.index,quantile_dist,'blue')
        axes[0,i].plot(quantile_low_2.index,quantile_low_2,'red')
        axes[0,i].axvline(dopage,color = 'black')
        
        
        #12 dichlor for radix dev.
        s = 'R'
        df = dfs[s]
        
        #mean treatment of data
        t_mins = 5
        df_mean = d_.rolling_mean(df,t_mins)
        
        #lower quantile
        quantile_distRAW = df.quantile(q = optimum, axis = 1)**2
        quantile_dist = d_.rolling_mean(quantile_distRAW,t_mins)
        
        low_cutoff = 2.34
        maxvalue = 6
        scale_factor = maxvalue/low_cutoff
        
        quantile_low_1 = df_mean.quantile(q = 1 - optimum, axis = 1)  #overestimate of quantile should use mean of raw values
        quantile_low_2 = d_.rolling_mean(df.quantile(q = 1 - optimum, axis = 1),t_mins)
        
        """
        IGT_low = quantile_low - low_cutoff
        IGT_low[IGT_low > 0] = 0.0
        IGT_low = -((IGT_low) * scale_factor)**2
        """
        
        #visualise plot
        axes[1,i].plot(quantile_dist.index,quantile_dist,'blue')
        axes[1,i].plot(quantile_low_2.index,quantile_low_2,'red')
        axes[1,i].axvline(dopage,color = 'black')