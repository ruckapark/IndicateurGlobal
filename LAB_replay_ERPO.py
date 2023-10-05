# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 09:49:58 2023

algorithm development for replayed data Erpobdella

@author: George
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from sklearn import preprocessing
from datetime import timedelta,datetime
from scipy import signal

#%% IMPORT personal mods
os.chdir('MODS')
from data_merge import merge_dfs
from dope_reg import dope_read_extend
import dataread as d_
os.chdir('..')

#%% Relevant directories
"""
roots = ['765_20211022',
         '762_20211022',
         '763_20211022',
         '762_20211028',
         '762_20220225',
         '763_20220225',
         '764_20220310',
         '765_20220310',
         '765_20220317',
         '760_20220708',
         '761_20220708',
         '762_20220708',
         '763_20220708',
         '764_20220708',
         '765_20220708',
         '766_20220708',
         '767_20220708',
         '768_20220708',
         '769_20220708']
"""
roots = ['769_20220610']
    

def plot_distribution(val1,val2,species = 'E',figname = None):
    xlims = {'E':1000,'G':1000,'R':200}
    
    plt.figure()
    sns.histplot(val1)
    sns.histplot(val2)
    plt.xlim(0,xlims[species])
    
    if figname: plt.savefig(r'C:\Users\George\Documents\Figures\DeepReplay\{}_{}_histogram.jpg'.format(species,figname))
    
    plt.figure()
    plt.plot(np.arange(0,1,0.01),np.array([np.quantile(val1,i/100) for i in range(100)]))
    plt.plot(np.arange(0,1,0.01),np.array([np.quantile(val2,i/100) for i in range(100)]))
    
    if figname: plt.savefig(r'C:\Users\George\Documents\Figures\DeepReplay\{}_{}_QuantilePlot.jpg'.format(species,figname))
    
def read_starttime(root):
    """ read datetime and convert to object from txt file """
    startfile = open(r'{}\start.txt'.format(root),"r")
    starttime = startfile.read()
    startfile.close()
    return datetime.strptime(starttime,'%d/%m/%Y %H:%M:%S')

def correct_index(df,start,correction = 0.997):
    """ Account for time warp error in video generation """
    ind = np.array((df.index - df.index[0]).total_seconds() * correction)
    ind = pd.to_datetime(ind*pd.Timedelta(1,unit = 's') + pd.to_datetime(start))
    df_ = df.copy()
    df_.index = ind
    return df_

def read_dead(directory):
    
    morts = {'E':[],'G':[],'R':[]}
    with open(r'{}\morts.csv'.format(directory)) as f:
        reader_obj = csv.reader(f)
        for row in reader_obj:
            try:
                int(row[1])
                morts[row[0][0]] = [int(x) for x in row[1:]]
            except ValueError:
                continue
    return morts

def filter_erpo(df_r,df,df_q):
    
    """
    df_ is the replayed version with corrected index
    df original
    df_q quantization original
    
    use Erpo specific algorithm to filter data and create new df
    """
    
    thresh_erpo = {
        'high':250,
        'mid':150,
        'low':100,
        'q':0.8
        }
    
    for col in df_r.columns:
        replay = df_r[col]
        old = df[col]
        
        """
        high outliers 
        check quantization to decided if noise or real high value
        """
        outliers_high = replay[replay > thresh_erpo['high']]
        for t in outliers_high.index:
            
            #surrounding quantizations from surrounding df
            qs = df_q[(df_q.index > t - pd.Timedelta(30,'s')) & (df_q.index < t + pd.Timedelta(30,'s'))][col]
            #get timestamp of highest value in quantization
            ind_old = qs.idxmax()
            
            #if max quantif around outliers below 0.8 or quantization threshold
            if qs[ind_old] < thresh_erpo['q']:
                
                #if old value is low use this
                if old.loc[ind_old] < thresh_erpo['low']:
                    df_r.loc[t][col] = old.loc[ind_old]
                #otherwise add 0.0
                else:
                    df_r.loc[t][col] = 0.0
                    
            #if high quantization value
            else:
                if old.loc[ind_old] < replay[t]:
                    df_r.loc[t][col] = old.loc[ind_old]
        
        replay = df_r[col]
        old = df[col]
                    
        """
        mid outliers 
        check for more reasonable values in original df
        """
        outliers_mid = replay[replay > thresh_erpo['mid']]
        for t in outliers_mid.index:
            
            #surrounding values from old df +- 20 seconds
            old_close = old[(df.index > t - pd.Timedelta(20,'s')) & (df.index < t + pd.Timedelta(20,'s'))]
            
            #if corresponding value less than low thresh
            if old_close.values.max() < df_r.loc[t][col]:
                df_r.loc[t][col] = old_close.values.max()
                    
    
    return df_r

#%% Code

"""
% of time higher then the other
average difference between values 
"""

if __name__ == '__main__':
    
    specie = {'E': 'Erpobdella','G':'Gammarus','R':'Radix'}
    time_correction = 0.997
    values_old,values_new = np.array([]),np.array([])
    
    for r in roots[0:1]:
        
        Tox = int(r.split('_')[0])
        
        stem = [d for d in os.listdir(r'I:\TXM{}-PC'.format(Tox)) if r.split('_')[-1] in d]
        root = r'I:\TXM{}-PC\{}'.format(Tox,stem[0])
        starttime = read_starttime(root)
        
        #locate original and copy
        file_og = r'{}\{}.xls.zip'.format(root,stem[0])
        file_copy = r'{}\{}.replay.xls.zip'.format(root,stem[0])
        mapping = d_.read_mapping(Tox,int(r.split('_')[-1]))
        
        #read file
        df_og,df_copy = d_.read_merge([file_og]),d_.read_merge([file_copy])
        dfs_og,dfs_copy = d_.preproc(df_og),d_.preproc(df_copy)
        dfs_og,dfs_copy = d_.calibrate(dfs_og,Tox,starttime),d_.calibrate(dfs_copy,Tox,starttime)
        
        #check mapping
        dfs_og = d_.check_mapping(dfs_og,mapping)
        
        #read dead values
        morts = read_dead(root)
        dfs_og,dfs_copy = d_.remove_dead_known(dfs_og,morts),d_.remove_dead_known(dfs_copy,morts)
        
        for s in specie:
            dfs_copy[s] = correct_index(dfs_copy[s], starttime, time_correction)
        
        species = 'E'
        df1,df2 = dfs_og[species],dfs_copy[species]
        df1_m,df2_m = d_.rolling_mean(df1,5),d_.rolling_mean(df2,5)
        
        #read in quantization and check for count of mid bursts
        df_quant_mid = d_.read_quant([file_og])
        df_q = d_.preproc(df_quant_mid,quant = True)[species]
        
        #check mapping
        df_q = d_.check_mapping(df_q,mapping[species])
        
        #%%
        t_ind1,t_ind2 = np.array((df1.index - df1.index[0]).total_seconds()),np.array((df2.index - df2.index[0]).total_seconds())
        tm_ind1,tm_ind2 = np.array((df1_m.index - df1_m.index[0]).total_seconds()),np.array((df2_m.index - df2_m.index[0]).total_seconds())
        
        fig,axe = plt.subplots(4,4,figsize = (12,20),sharex = True)
        axe_q = np.empty(axe.shape,dtype = object)
        for i in range(16):
            axe_q[i//4,i%4] = axe[i//4,i%4].twinx()
            if i+1 not in df1.columns: continue
            axe[i//4,i%4].plot(t_ind1,df1[i+1])
            axe[i//4,i%4].plot(t_ind2,df2[i+1])
            axe_q[i//4,i%4].plot(t_ind1,df_q[i+1],color = 'r',alpha = 0.3)
        fig.tight_layout()
        
        
        #%% Start threshold treatments
        thresh_high = 250
        thresh_mid = 150
        thresh_low = 100
        thresh_q = 0.8
        for i in df2.columns:
            replay = df2[i]
            old = df1[i]
            outliers_high = replay[replay > thresh_high]
            
            ## loop high outliers
            for t in outliers_high.index:
                
                #surrounding timestamps from old df +- 30 seconds
                qs = df_q[(df_q.index > t - pd.Timedelta(30,'s')) & (df_q.index < t + pd.Timedelta(30,'s'))][i]
                
                #get timestamp of highest value in quantization
                ind_old = qs.idxmax()
                
                #if max quant around outliers below threshold
                if qs[ind_old] < thresh_q:
                    
                    #if old value is low use this
                    if old.loc[ind_old] < thresh_low:
                        df2.loc[t][i] = old.loc[ind_old]
                    #otherwise add 0.0
                    else:
                        df2.loc[t][i] = 0.0
                        
                #if high quantization value
                else:
                    if old.loc[ind_old] < replay[t]:
                        df2.loc[t][i] = old.loc[ind_old]
                        
            print('{} high values filtered in column: {}'.format(outliers_high.shape[0],i))
            
            replay = df2[i]
            old = df1[i]
            outliers_mid = replay[replay > thresh_mid]
            
            ## loop mid outliers
            for t in outliers_mid.index:
                
                #surrounding values from old df +- 20 seconds
                old_close = old[(df1.index > t - pd.Timedelta(20,'s')) & (df1.index < t + pd.Timedelta(20,'s'))]
                
                #if corresponding value less than low thresh
                if old_close.values.max() < df2.loc[t][i]:
                    df2.loc[t][i] = old_close.values.max()
                
            print('{} mid values filtered in column: {}'.format(outliers_mid.shape[0],i))
            
            
        # low outliers - check surrounding values for high quantile 95, if so replace with old value
        
        fig,axe = plt.subplots(4,4,figsize = (12,20),sharex = True)
        for i in range(16):
            if i+1 not in df1.columns: continue
            axe[i//4,i%4].plot(t_ind1,df1[i+1])
            axe[i//4,i%4].plot(t_ind2,df2[i+1],color = 'red',alpha = 0.75)
        fig.tight_layout()
        
        #%%
        filter_erpo(df2,df1,df_q)