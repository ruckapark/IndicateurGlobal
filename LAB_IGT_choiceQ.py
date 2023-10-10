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
    
    quantiles = [0.05,0.1,0.15,0.25,0.5,0.75,0.85,0.9,0.95]
    low_q,mid_q,high_q = [0.5,0.1,0.15],[0.25,0.5,0.75],[0.85,0.9,0.95]
    #quantiles = [0.1] #debug
    controldata = {s:{q:np.array([],dtype = float) for q in quantiles} for s in specie}
    
    #collect relevant roots eventually looping through species
    for s in specie:
        for Tox in range(760,769):
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
            df = df[df.index < dopage - pd.Timedelta(hours = 1)]
            
            for quantile in quantiles:
                quantdata = np.array(df.quantile(q = quantile, axis = 1))
                controldata[s][quantile] = np.hstack((controldata[s][quantile],quantdata))
    
    #%% plot histograms
    plt.close('all')
    sns.set_style("white")
    
    #plot histograms
    for s in specie:
        for quantile in low_q:
            ratio_q = np.sum(controldata[s][quantile] < 1) / controldata[s][quantile].shape[0]
            histdata = controldata[s][quantile][controldata[s][quantile] > 1]
            fig = plt.figure(figsize = (12,7))
            axe = fig.add_axes([0.1,0.1,0.8,0.8])
            axe.set_title('{} Control Distiburion: Q = {}   -   {:.2f}% values below 1.0'.format(specie[s],quantile,100*ratio_q))
            sns.histplot(histdata,ax=axe)
            
        for quantile in mid_q:
            histdata = controldata[s][quantile]
            mean,median,std = np.mean(histdata),np.median(histdata),np.std(histdata)
            fig = plt.figure(figsize = (12,7))
            axe = fig.add_axes([0.1,0.1,0.8,0.8])
            axe.set_title('{} Control Distiburion: Q = {}   -   Mean{:.2f}  Median{:.2f}  Std{:.2f}'.format(specie[s],quantile,mean,median,std))
            sns.histplot(histdata,ax=axe)
            
        for quantile in high_q:
            histdata = controldata[s][quantile]
            mean,median,std = np.mean(histdata),np.median(histdata),np.std(histdata)
            fig = plt.figure(figsize = (12,7))
            axe = fig.add_axes([0.1,0.1,0.8,0.8])
            axe.set_title('{} Control Distiburion: Q = {}   -   Mean{:.2f}  Median{:.2f}  Std{:.2f}'.format(specie[s],quantile,mean,median,std))
            if s != 'G':
                axe.axvline(mean - 1.5*std,color = 'r')
                axe.axvline(mean + 1.5*std,color = 'r')
            sns.histplot(histdata,ax=axe)