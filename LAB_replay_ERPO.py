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
    
    for r in roots[2:3]:
        
        Tox = int(r.split('_')[0])
        
        stem = [d for d in os.listdir(r'I:\TXM{}-PC'.format(Tox)) if r.split('_')[-1] in d]
        root = r'I:\TXM{}-PC\{}'.format(Tox,stem[0])
        starttime = read_starttime(root)
        
        #locate original and copy
        file_og = r'{}\{}.xls.zip'.format(root,stem[0])
        file_copy = r'{}\{}.replay.xls.zip'.format(root,stem[0])
        
        #read file
        df_og,df_copy = d_.read_merge([file_og]),d_.read_merge([file_copy])
        dfs_og,dfs_copy = d_.preproc(df_og),d_.preproc(df_copy)
        dfs_og,dfs_copy = d_.calibrate(dfs_og,Tox,starttime),d_.calibrate(dfs_copy,Tox,starttime)
        
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
        
        #%% Start with Radix
        
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
        
        
        
        
        
        """
        fig,axe = plt.subplots(4,4,figsize = (12,20),sharex = True,sharey = True)
        for i in range(16):
            if i+1 not in df1.columns: continue
            axe[i//4,i%4].plot(tm_ind1,df1_m[i+1])
            axe[i//4,i%4].plot(tm_ind2,df2_m[i+1])
            axe[i//4,i%4].set_ylim([0,800])
        fig.tight_layout()
        
        #%% Quantization
        
        fig,axe = plt.subplots(4,4,figsize = (12,20),sharex = True,sharey = True)
        df_qm = d_.rolling_mean(df_q,20)
        tq = (df_qm.index - df_qm.index[0]).total_seconds()
        for i in range(16):
            if i+1 not in df_qm.columns: continue
            axe[i//4,i%4].plot(tq,df_qm[i+1])
        fig.tight_layout()
        
        
        #periods of high activity
        high_activity = {
            1:[0,100000],
            2:[0,100000],
            3:[0,60000],
            4:[40000,60000],
            5:[43000,60000],
            6:[0,40000],
            7:[0,55000],
            9:[0,100000],
            11:[0,35000],
            12:[0,60000],
            13:[30000,40000],
            14:[65000,100000],
            15:[0,120000],
            16:[40000,120000]}
        
        for col in high_activity:
            start_ = starttime + pd.Timedelta(high_activity[col][0],unit = 's')
            end_ = starttime + pd.Timedelta(high_activity[col][1],unit = 's')
            
            og,copy = df1[col][(df1.index > start_) & (df1.index < end_)],df2[col][(df2.index > start_) & (df2.index < end_)]
            
            print('Original quantile: {}'.format(int(og.quantile(0.95))),'Copied quantile: {}'.format(int(copy.quantile(0.95))))
        """