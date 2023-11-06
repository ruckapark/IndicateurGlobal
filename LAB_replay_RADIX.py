# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 11:51:51 2023

Compare results of datafiles from the replay versions of files

@author: George
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
# roots = ['765_20211022',
#          '767_20211022',
#          '762_20211022',
#          '763_20211022',
#          '762_20211028',
#          '762_20220225',
#          '763_20220225',
#          '764_20220310',
#          '765_20220310',
#          '765_20220317',
#          '760_20220708',
#          '761_20220708',
#          '762_20220708',
#          '763_20220708',
#          '764_20220708',
#          '765_20220708',
#          '766_20220708',
#          '767_20220708',
#          '768_20220708',
#          '769_20220708']

roots = ['762_20210430','762_20210506','762_20210513','761_20210506','760_20210430','760_20210506','760_20210513']

def plot_distribution(val1 = None,val2 = None,species = 'R',figname = None):
    xlims = {'E':1000,'G':1000,'R':200}
    
    plt.figure()
    sns.histplot(val1)
    sns.histplot(val2)
    plt.xlim(0,xlims[species])
    
    if figname: plt.savefig(r'C:\Users\George\Documents\Figures\DeepReplay\{}_{}_histogram.jpg'.format(species,figname))
    
    plt.figure()
    if type(val1) == np.ndarray: plt.plot(np.arange(0,1,0.01),np.array([np.quantile(val1,i/100) for i in range(100)]))
    if type(val2) == np.ndarray: plt.plot(np.arange(0,1,0.01),np.array([np.quantile(val2,i/100) for i in range(100)]))
    
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

def filter_radix(df_r,df = None):
    
    """
    df_ is the replayed version with corrected index
    df original
    
    use Radix specific algorithm to filter data and create new df
    """
    
    thresh_radix = {
        'high':33,
        'mid':21,
        'low':10
        }
    
    if type(df) == pd.core.frame.DataFrame:
        for col in df_r.columns:
            replay = df_r[col]
            old = df[col]
            outliers = replay[replay > thresh_radix['high']]
            
            for t in outliers.index:
                ind_old = old[(old.index > t - pd.Timedelta(11,'s')) & (old.index < t + pd.Timedelta(11,'s'))].index[0]
                if old.loc[ind_old] < thresh_radix['mid']:
                    df_r.loc[t][col] = old.loc[ind_old]
                else:
                    df_r.loc[t][col] = 0.0
    
    else:
        
        #high outliers
        for col in df_r.columns:
            replay = df_r[col]
            outliers = replay[replay > thresh_radix['high']]
            
            for t in outliers.index:
                
                #surrounding values
                ds = replay[(replay.index > t - pd.Timedelta(120,'s')) & (replay.index < t + pd.Timedelta(120,'s'))]
                
                vals = ds.values
                vals.sort()
                
                if vals.shape[0]:
                    #if half values v low, remove
                    if np.mean(vals[:vals.shape[0]//2]) < 2:
                        df_r.loc[t][i] = 0.0
                
                    #if most values much lower, replace
                    elif np.mean(vals[:3*(vals.shape[0]//4)]) < thresh_radix['low']:
                        df_r.loc[t][i] = np.mean(vals[:3*(vals.shape[0]//4)])
                
                else: 
                    continue
            
        
            #mid outliers
            replay = df_r[col]
            outliers_mid = replay[replay > thresh_radix['mid']]
            
            for t in outliers_mid.index:
                
                #surrounding values from old df +- 20 seconds
                ds = replay[(replay.index > t - pd.Timedelta(120,'s')) & (replay.index < t + pd.Timedelta(120,'s'))]
                
                vals = ds.values
                vals.sort()
                
                if vals.shape[0]:
                    #if half values v low, remove
                    if np.mean(vals[:3*(vals.shape[0]//4)]) < 1:
                        df_r.loc[t][i] = 0.0
                
                    #if most values much lower, replace
                    elif np.mean(vals[:3*(vals.shape[0]//4)]) < thresh_low:
                        df_r.loc[t][i] = np.mean(vals[:3*(vals.shape[0]//4)])
                
                else: 
                    continue
    
    return df_r

#%% Code

"""
% of time higher then the other
average difference between values 
"""

if __name__ == '__main__':
    
    specie = {'E': 'Erpobdella','G':'Gammarus','R':'Radix'}
    time_correction = 0.997
    thresh_ = [[],[]]
    values_old,values_new = np.array([]),np.array([])
    
    for r in roots[:2]:
        
        Tox = int(r.split('_')[0])
        
        stem = [d for d in os.listdir(r'I:\TXM{}-PC'.format(Tox)) if r.split('_')[-1] in d]
        root = r'I:\TXM{}-PC\{}'.format(Tox,stem[0])
        starttime = read_starttime(root)
        
        #locate original and copy
        file_og = r'{}\{}.xls.zip'.format(root,stem[0])
        if not os.path.isfile(file_og): file_og = None
        file_copy = r'{}\{}.replay.xls.zip'.format(root,stem[0])
        mapping = d_.read_mapping(Tox,int(r.split('_')[-1]))
        
        #read file
        if file_og:
            df_og = d_.read_merge([file_og])
            dfs_og = d_.preproc(df_og)
            dfs_og = d_.calibrate(dfs_og,Tox,starttime)
        
        df_copy = d_.read_merge([file_copy])
        dfs_copy = d_.preproc(df_copy)
        dfs_copy = d_.calibrate(dfs_copy,Tox,starttime)
        
        for s in specie:
            dfs_copy[s] = correct_index(dfs_copy[s], starttime, time_correction)
        
        species = 'R'
        
        
        
        
        ## If original file exists
        
        
        
        if file_og:
            
            df1,df2 = dfs_og[species],dfs_copy[species]
            df1_m,df2_m = d_.rolling_mean(df1,5),d_.rolling_mean(df2,5)
            
            
            #%% Start with Radix
            
            t_ind1,t_ind2 = np.array((df1.index - df1.index[0]).total_seconds()),np.array((df2.index - df2.index[0]).total_seconds())
            tm_ind1,tm_ind2 = np.array((df1_m.index - df1_m.index[0]).total_seconds()),np.array((df2_m.index - df2_m.index[0]).total_seconds())
            
            fig,axe = plt.subplots(4,4,figsize = (12,20),sharex = True)
            for i in range(16):
                axe[i//4,i%4].plot(t_ind1,df1[i+1])
                axe[i//4,i%4].plot(t_ind2,df2[i+1])
            fig.tight_layout()
            
            fig,axe = plt.subplots(4,4,figsize = (12,20),sharex = True,sharey = True)
            for i in range(16):
                axe[i//4,i%4].plot(tm_ind1,df1_m[i+1])
                axe[i//4,i%4].plot(tm_ind2,df2_m[i+1])
                axe[i//4,i%4].set_ylim([0,800])
            fig.tight_layout()
            
            #plot distribution comparison
            values1,values2 = df1.values.flatten(),df2.values.flatten()
            values1,values2 = values1[values1 > 0],values2[values2 > 0]
            
            values_old,values_new = np.hstack((values_old,values1)),np.hstack((values_new,values2))
            
            plot_distribution(values1,values2,species,figname = r)
            
            #two thresholds, one for evaluation and one for direct cut
            thresh_bur = np.quantile(values2,0.98)
            thresh_mid = np.quantile(values2,0.97) #could also base this on organism size as well
            thresh_[0].append(thresh_bur),thresh_[1].append(thresh_mid)
            
            
        # Else if only replay file
        
        
        else:
            
            df2 = dfs_copy[species]
            df2_m = d_.rolling_mean(df2,5)
            
            
            #%% Start with Radix
            
            t_ind2 = np.array((df2.index - df2.index[0]).total_seconds())
            tm_ind2 = np.array((df2_m.index - df2_m.index[0]).total_seconds())
            
            fig,axe = plt.subplots(4,4,figsize = (12,20),sharex = True)
            for i in range(16):
                axe[i//4,i%4].plot(t_ind2,df2[i+1])
            fig.tight_layout()
            
            fig,axe = plt.subplots(4,4,figsize = (12,20),sharex = True,sharey = True)
            for i in range(16):
                axe[i//4,i%4].plot(tm_ind2,df2_m[i+1])
                axe[i//4,i%4].set_ylim([0,800])
            fig.tight_layout()
            
            #plot distribution comparison
            values2 = df2.values.flatten()
            values2 = values2[values2 > 0]
            
            values_old = None
            values_new = np.hstack((values_new,values2))
            
            plot_distribution(val2 = values2,species = species)
            
            #two thresholds, one for evaluation and one for direct cut
            thresh_bur = np.quantile(values2,0.98)
            thresh_mid = np.quantile(values2,0.97) #could also base this on organism size as well
            thresh_[0].append(thresh_bur),thresh_[1].append(thresh_mid)
        
        
    plot_distribution(values_old,values_new,species)
    
    
    #%%
    thresh_high = 33
    thresh_mid = 21
    thresh_low = 10
    
    if file_og:
    
        for i in df2.columns:
            replay = df2[i]
            old = df1[i]
            outliers = replay[replay > thresh_high]
            
            for t in outliers.index:
                ind_old = old[(old.index > t - pd.Timedelta(11,'s')) & (old.index < t + pd.Timedelta(11,'s'))].index[0]
                if old.loc[ind_old] < thresh_mid:
                    df2.loc[t][i] = old.loc[ind_old]
                else:
                    df2.loc[t][i] = 0.0
                    
        fig,axe = plt.subplots(4,4,figsize = (12,20),sharex = True)
        for i in range(16):
            axe[i//4,i%4].plot(t_ind1,df1[i+1])
            axe[i//4,i%4].plot(t_ind2,df2[i+1])
        fig.tight_layout()
        
        df2__m = d_.rolling_mean(df2,5)
        #%%
        fig,axe = plt.subplots(4,4,figsize = (12,20),sharex = True)
        for i in range(16):
            axe[i//4,i%4].plot(tm_ind1,df1_m[i+1])
            axe[i//4,i%4].plot(tm_ind2,df2__m[i+1],color = 'r',linestyle = '--',alpha = 0.5)
            axe[i//4,i%4].set_ylim([0,80])
        fig.tight_layout()
        
    else:
        
        fig,axe = plt.subplots(4,4,figsize = (12,20),sharex = True)
        for i in range(16):
            axe[i//4,i%4].plot(t_ind2,df2[i+1])
        
        for i in df2.columns:
            
            #high outliers
            replay = df2[i]
            outliers = replay[replay > thresh_high]
            
            for t in outliers.index:
                
                #surrounding values
                ds = replay[(replay.index > t - pd.Timedelta(120,'s')) & (replay.index < t + pd.Timedelta(120,'s'))]
                
                vals = ds.values
                vals.sort()
                
                if vals.shape[0]:
                    #if half values v low, remove
                    if np.mean(vals[:vals.shape[0]//2]) < 2:
                        df2.loc[t][i] = 0.0
                
                    #if most values much lower, replace
                    elif np.mean(vals[:3*(vals.shape[0]//4)]) < thresh_low:
                        df2.loc[t][i] = np.mean(vals[:3*(vals.shape[0]//4)])
                
                else: 
                    continue
                
            
            #mid outliers
            replay = df2[i]
            outliers_mid = replay[replay > thresh_mid]
            
            for t in outliers_mid.index:
                
                #surrounding values from old df +- 20 seconds
                ds = replay[(replay.index > t - pd.Timedelta(120,'s')) & (replay.index < t + pd.Timedelta(120,'s'))]
                
                vals = ds.values
                vals.sort()
                
                if vals.shape[0]:
                    #if half values v low, remove
                    if np.mean(vals[:3*(vals.shape[0]//4)]) < 1:
                        df2.loc[t][i] = 0.0
                
                    #if most values much lower, replace
                    elif np.mean(vals[:3*(vals.shape[0]//4)]) < thresh_low:
                        df2.loc[t][i] = np.mean(vals[:3*(vals.shape[0]//4)])
                
                else: 
                    continue
            
        if file_og:         
            fig,axe = plt.subplots(4,4,figsize = (12,20),sharex = True)
            for i in range(16):
                axe[i//4,i%4].plot(t_ind1,df1[i+1])
                axe[i//4,i%4].plot(t_ind2,df2[i+1])
            fig.tight_layout()
            
            df2__m = d_.rolling_mean(df2,5)
            #%%
            fig,axe = plt.subplots(4,4,figsize = (12,20),sharex = True)
            for i in range(16):
                axe[i//4,i%4].plot(tm_ind1,df1_m[i+1])
                axe[i//4,i%4].plot(tm_ind2,df2__m[i+1],color = 'r',linestyle = '--',alpha = 0.5)
                axe[i//4,i%4].set_ylim([0,80])
            fig.tight_layout()
        else:
            
            for i in range(16):
                axe[i//4,i%4].plot(t_ind2,df2[i+1],color = 'red')
                
        temp = filter_radix(df2)