# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 09:37:54 2023

Use pre dopage data to decide on appropriate quantiles

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

import LAB_readcsv as CSV_

#%% main code
if __name__ == '__main__':
    
    plt.close('all')
    
    specie = {'E': 'Erpobdella','G':'Gammarus','R':'Radix'}
    directories = {s:[] for s in specie}
    
    quantiles = [0.0667,0.12917,0.19091,0.25,0.5,0.75,0.80909,0.87083,0.9333]
    low_q,mid_q,high_q = [0.0667,0.12917,0.19091],[0.25,0.5,0.75],[0.80909,0.87083,0.9333]
    #quantiles = [0.1] #debug
    controldata = {s:{q:np.array([],dtype = float) for q in quantiles} for s in specie}
    spikedata = {s:{q:np.array([],dtype = float) for q in quantiles} for s in specie}
    
    #collect relevant roots eventually looping through species
    for s in specie:
        for Tox in range(760,769):
            if Tox == 766: continue #temoin
            base = r'I:\TXM{}-PC'.format(Tox)
            for d in [direc for direc in os.listdir(base) if os.path.isdir(r'{}\{}'.format(base,direc))]:
                filename = '{}_{}.csv.zip'.format(d.split('-')[0],specie[s])
                if os.path.isfile(r'{}\{}\{}'.format(base,d,filename)):
                    directories[s].append(r'{}\{}\{}'.format(base,d,filename))
    
    dope_df = dope_read_extend()
    
    for s in specie:
        for file in directories[s]:
            
            root = r'{}\{}\{}'.format(file.split('\\')[:-1][0],file.split('\\')[:-1][1],file.split('\\')[:-1][2])
            
            try:
                dopage_entry = CSV_.find_dopage_entry(dope_df, root)
                dopage = dopage_entry['Start']
            except:
                continue
        
            try:
                morts = d_.read_dead(root)
            except:
                continue
            
            if len(morts[s]) > 11:
                continue
                
            #read in dataframe
            df = pd.read_csv(file,index_col = 0)
            df = CSV_.preproc_csv(df)
            
            rewrite_csv = False
            for m in morts[s]:
                if m in df.columns: 
                    df = df.drop(columns = [m])
                    rewrite_csv = True
            if rewrite_csv: 
                d_.write_csv(df,s,root)
                
            #select all data from previous to one hour before dopage
            df_control = df[(df.index > dopage - pd.Timedelta(hours = 12)) & (df.index < dopage - pd.Timedelta(hours = 0.5))]
            df_spike = df[(df.index > dopage) & (df.index < dopage + pd.Timedelta(hours = 4))]
            
            for quantile in quantiles:
                
                #control data
                quantdata = np.array(df_control.quantile(q = quantile, axis = 1))
                controldata[s][quantile] = np.hstack((controldata[s][quantile],quantdata))
                
                #spike data
                quantdata_spike = np.array(df_spike.quantile(q = quantile, axis = 1))
                spikedata[s][quantile] = np.hstack((spikedata[s][quantile],quantdata_spike))
    
    #%% plot histograms
    plt.close('all')
    sns.set_style("white")
    
    #plot histograms
    for s in specie:
        
        #lower quantiles
        for quantile in low_q:
            
            #control
            ratio_q = np.sum(controldata[s][quantile] < 1) / controldata[s][quantile].shape[0]
            histdata = controldata[s][quantile][controldata[s][quantile] > 1]
            
            #spike data
            ratio_q_spike = np.sum(spikedata[s][quantile] < 1) / spikedata[s][quantile].shape[0]
            histdata_spike = spikedata[s][quantile][spikedata[s][quantile] > 1]
            
            fig,axes = plt.subplots(1,2,figsize = (16,7),sharex = True)
            plt.suptitle('{} Control Distiburion: Q = {}'.format(specie[s],quantile))
            
            #control
            sns.histplot(histdata,ax=axes[0])
            axes[0].set_title('{:.2f}% values below 1.0'.format(100*ratio_q))
            
            #spike
            sns.histplot(histdata_spike,ax=axes[1])
            axes[1].set_title('{:.2f}% values below 1.0'.format(100*ratio_q_spike)) #spike
        
            
        #mid quantiles
        for quantile in mid_q:
            
            histdata = controldata[s][quantile]
            mean,median,std = np.mean(histdata),np.median(histdata),np.std(histdata)
            
            histdata_spike = spikedata[s][quantile]
            mean_spike,median_spike,std_spike = np.mean(histdata_spike),np.median(histdata_spike),np.std(histdata_spike)
            
            fig,axes = plt.subplots(1,2,figsize = (16,7),sharex = True)
            plt.suptitle('{} Control Distiburion: Q = {}'.format(specie[s],quantile))
            
            #control
            sns.histplot(histdata,ax=axes[0])
            axes[0].set_title('Mean{:.2f}  Median{:.2f}  Std{:.2f}'.format(mean,median,std))
            
            #spike
            sns.histplot(histdata_spike,ax=axes[1])
            axes[1].set_title('Mean{:.2f}  Median{:.2f}  Std{:.2f}'.format(mean_spike,median_spike,std_spike))
        
        
        #high quantiles
        for quantile in high_q:
            
            histdata = controldata[s][quantile]
            mean,median,std = np.mean(histdata),np.median(histdata),np.std(histdata)
            
            histdata_spike = spikedata[s][quantile]
            mean_spike,median_spike,std_spike = np.mean(histdata_spike),np.median(histdata_spike),np.std(histdata_spike)
            
            fig,axes = plt.subplots(1,2,figsize = (16,7),sharex = True)
            plt.suptitle('{} Control Distiburion: Q = {}'.format(specie[s],quantile))
            
            if s != 'G':
                for i in range(2):
                    axes[i].axvline(mean - 1.5*std,color = 'black')
                    axes[i].axvline(mean + 1.5*std,color = 'black')
                    
                    axes[i].axvline(mean - 2*std,color = 'black',linestyle = '--')
                    axes[i].axvline(mean + 2*std,color = 'black',linestyle = '--')
                
            #control
            sns.histplot(histdata,ax=axes[0])
            axes[0].set_title('Mean{:.2f}  Median{:.2f}  Std{:.2f}'.format(mean,median,std))
            
            #spike
            sns.histplot(histdata_spike,ax=axes[1])
            axes[1].set_title('Mean{:.2f}  Median{:.2f}  Std{:.2f}'.format(mean_spike,median_spike,std_spike))