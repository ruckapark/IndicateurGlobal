# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 11:20:36 2023

Use the replay filter functions from LAB_replay_//
-timelag for Gammarus
-ERPO for Erpobdella
-RADIX for Radix

Check results are coherent

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

#%% IMPORT replay files
from LAB_replay_ERPO import filter_erpo
from LAB_replay_timelag import filter_gammarus
from LAB_replay_RADIX import filter_radix

#time corrections
exceptions = {
    r'E_I:\TXM761-PC\20210506-230122':2.493,
    r'E_I:\TXM760-PC\20210513-230436':2.493
    }

#%% Function to write to directory (and zip)

def read_roots():
    
    file1 = open(r'D:\VP\Viewpoint_data\replaydata.txt', 'r')
    Lines = file1.readlines()
    return [r.split(',')[0][:-1] for r in Lines]  #-1 is short fix for function read \n at end of every line

if __name__ == "__main__":
    
    #read files to be studied
    roots = read_roots()
    failed = []
    nodead = []
    preprocessfailure = []
    datafailure = []
    
    specie = {'E': 'Erpobdella','G':'Gammarus','R':'Radix'}
    
    #debug
    for r in [r'I:\TXM760-PC\20211003-010154']:
    #for r in roots:
        
        plt.close('all')
    
        time_correction = 0.997
        values_old,values_new = np.array([]),np.array([])
        
        replay_data = {}
        
        Tox = int(r.split('\\')[1].split('-')[0][3:])
        #stem = [d for d in os.listdir(r'I:\TXM{}-PC'.format(Tox)) if r.split('_')[-1] in d]
        stem = r.split('\\')[-1]
        root = r
        mapping = d_.read_mapping(Tox) #if mapping date necessary: int(r.split('\\')[-1].split('-')[0])
        
        #account for false 769 data
        remapping = None
        if Tox == 769: remapping = d_.read_mapping(Tox,remapping = True)
        
        try:
            starttime = d_.read_starttime(root)
        except:
            failed.append(r)
            continue
        
        #read old and new xls file - what if the old file no longer exists? simplify functions
        file_og = r'{}\{}.xls.zip'.format(root,stem)
        if not os.path.isfile(file_og): file_og = None
        file_copy = r'{}\{}.replay.xls.zip'.format(root,stem)
        
        #read file, wrangle and calibrate
        if file_og:
            try:
                df_og = d_.read_merge([file_og])
                dfs_og = d_.preproc(df_og)
                dfs_og = d_.calibrate(dfs_og,Tox,starttime)
                dfs_og = d_.check_mapping(dfs_og,mapping)
            except:
                #swap retreat algorithm to new when only recent exists
                preprocessfailure.append(r)
                continue
            
        try:
            df_copy = d_.read_merge([file_copy])
            dfs_copy = d_.preproc(df_copy)
            dfs_copy = d_.calibrate(dfs_copy,Tox,starttime)
            if remapping: dfs_copy = d_.check_mapping(dfs_copy,remapping)
        
            #correct time index including time warp, original if necessary
            reset_original = False
            if file_og:
                start_ind = None
                if dfs_og[[*dfs_og][0]].shape[0]:
                    start_ind = dfs_og[[*dfs_og][0]].index[0]
                elif dfs_og[[*dfs_og][1]].shape[0]:
                    start_ind = dfs_og[[*dfs_og][1]].index[0]
                elif dfs_og[[*dfs_og][2]].shape[0]:
                    start_ind = dfs_og[[*dfs_og][2]].index[0]
                else:
                    preprocessfailure.append(r)
                    continue
                
                if datetime.strptime(str(start_ind).split(' ')[0],'%Y-%m-%d') == starttime.replace(hour=0, minute=0, second=0):
                    reset_original = False
                else:
                    reset_original = True
                    
            for s in specie:
                if r'{}_{}'.format(s,root) in exceptions:
                    #Note that there will be a different number of data points
                    print('Time Warp Exception found')
                    dfs_copy[s] = d_.correct_index(dfs_copy[s], starttime, correction = exceptions[r'{}_{}'.format(s,root)])
                    dfs_copy[s] = dfs_copy[s]/exceptions[r'{}_{}'.format(s,root)]
                else:
                    dfs_copy[s] = d_.correct_index(dfs_copy[s], starttime)
                if reset_original: d_.correct_index(dfs_og[s], starttime, correction = 1)
            
        except:
            preprocessfailure.append(r)
            continue
            
        #read dead values
        try:
            morts = d_.read_dead(root)
            if file_og: dfs_og = d_.remove_dead_known(dfs_og,morts)
            dfs_copy = d_.remove_dead_known(dfs_copy,morts)
        except:
            nodead.append(r)
            continue
        
        
        #%% Gammarus
        if file_og:
            try:
                species = 'G'
                df1,df2 = dfs_og[species],dfs_copy[species]
                
                #get seconds time indexes and plot original graph with quant figure
                t_ind1,t_ind2 = np.array((df1.index - df1.index[0]).total_seconds()),np.array((df2.index - df2.index[0]).total_seconds())
                fig,axe = plt.subplots(4,4,figsize = (12,20),sharex = True)
                axe_q = np.empty(axe.shape,dtype = object)
                for i in range(16):
                    axe_q[i//4,i%4] = axe[i//4,i%4].twinx()
                    if i+1 not in df1.columns: continue
                    axe[i//4,i%4].plot(t_ind1,df1[i+1])
                    axe[i//4,i%4].plot(t_ind2,df2[i+1])
                fig.tight_layout()
                
                fig.savefig(r'{}\G_rawdata.jpg'.format(r))
                
                df_r = filter_gammarus(df2,df1)
                replay_data.update({species:df_r.copy()})
                
                #plot amended time series
                fig,axe = plt.subplots(4,4,figsize = (12,20),sharex = True)
                for i in range(16):
                    if i+1 not in df1.columns: continue
                    axe[i//4,i%4].plot(t_ind1,df1[i+1])
                    axe[i//4,i%4].plot(t_ind2,df_r[i+1],color = 'red',alpha = 0.75)
                fig.tight_layout()
                
                fig.savefig(r'{}\G_before_after.jpg'.format(r))
                
                #%% Erpobdella
                species = 'E'
                df1,df2 = dfs_og[species],dfs_copy[species]
                
                #read in quantization and check for count of mid bursts
                df_quant_mid = d_.read_quant([file_og])
                df_q = d_.preproc(df_quant_mid,quant = True)[species]
                df_q = d_.check_mapping(df_q,mapping[species])
                
                #get seconds time indexes and plot original graph with quant figure
                t_ind1,t_ind2 = np.array((df1.index - df1.index[0]).total_seconds()),np.array((df2.index - df2.index[0]).total_seconds())
                fig,axe = plt.subplots(4,4,figsize = (12,20),sharex = True)
                axe_q = np.empty(axe.shape,dtype = object)
                for i in range(16):
                    axe_q[i//4,i%4] = axe[i//4,i%4].twinx()
                    if i+1 not in df1.columns: continue
                    axe[i//4,i%4].plot(t_ind1,df1[i+1])
                    axe[i//4,i%4].plot(t_ind2,df2[i+1])
                    axe_q[i//4,i%4].plot(t_ind1,df_q[i+1],color = 'r',alpha = 0.3)
                fig.tight_layout()
                
                fig.savefig(r'{}\E_rawdata.jpg'.format(r))
                
                df_r = filter_erpo(df2,df1,df_q)
                replay_data.update({species:df_r.copy()})
                
                #plot amended time series
                fig,axe = plt.subplots(4,4,figsize = (12,20),sharex = True)
                for i in range(16):
                    if i+1 not in df1.columns: continue
                    axe[i//4,i%4].plot(t_ind1,df1[i+1])
                    axe[i//4,i%4].plot(t_ind2,df_r[i+1],color = 'red',alpha = 0.75)
                fig.tight_layout()
                
                fig.savefig(r'{}\E_before_after.jpg'.format(r))
                
                #%% Radix
                species = 'R'
                df1,df2 = dfs_og[species],dfs_copy[species]
                
                #get seconds time indexes and plot original graph with quant figure
                t_ind1,t_ind2 = np.array((df1.index - df1.index[0]).total_seconds()),np.array((df2.index - df2.index[0]).total_seconds())
                fig,axe = plt.subplots(4,4,figsize = (12,20),sharex = True)
                axe_q = np.empty(axe.shape,dtype = object)
                for i in range(16):
                    axe_q[i//4,i%4] = axe[i//4,i%4].twinx()
                    if i+1 not in df1.columns: continue
                    axe[i//4,i%4].plot(t_ind1,df1[i+1])
                    axe[i//4,i%4].plot(t_ind2,df2[i+1])
                fig.tight_layout()
                
                fig.savefig(r'{}\R_rawdata.jpg'.format(r))
                
                df_r = filter_radix(df2,df1)
                replay_data.update({species:df_r.copy()})
                
                #plot amended time series
                fig,axe = plt.subplots(4,4,figsize = (12,20),sharex = True)
                for i in range(16):
                    if i+1 not in df1.columns: continue
                    axe[i//4,i%4].plot(t_ind1,df1[i+1])
                    axe[i//4,i%4].plot(t_ind2,df_r[i+1],color = 'red',alpha = 0.75)
                fig.tight_layout()
                
                fig.savefig(r'{}\R_before_after.jpg'.format(r))
                
                #%% write files and zip
                for s in replay_data:
                    filename = '{}_{}'.format(stem.split('-')[0],specie[s])
                    if os.path.isfile(r'{}\{}.zip'.format(root,filename)):
                        print('Files already exist')
                        break
                    compression_options = dict(method='zip', archive_name='{}.csv'.format(filename))
                    replay_data[s].to_csv(r'{}\{}.csv.zip'.format(root,filename))
            except:
                datafailure.append(r)
                continue
        
        else:
            try:
                
                species = 'G'
                df2 = dfs_copy[species]
                
                #get seconds time indexes and plot original graph with quant figure
                t_ind2 = np.array((df2.index - df2.index[0]).total_seconds())
                fig,axe = plt.subplots(4,4,figsize = (12,20),sharex = True)
                for i in range(16):
                    if i+1 not in df2.columns: continue
                    axe[i//4,i%4].plot(t_ind2,df2[i+1])
                
                df_r = filter_gammarus(df2)
                replay_data.update({species:df_r.copy()})
                
                #plot amended time series
                for i in range(16):
                    if i+1 not in df2.columns: continue
                    axe[i//4,i%4].plot(t_ind2,df_r[i+1],color = 'red',alpha = 0.75)
                fig.tight_layout()
                
                fig.savefig(r'{}\G_before_after.jpg'.format(r))
                
                #%% Erpobdella
                species = 'E'
                df2 = dfs_copy[species]
                
                #get seconds time indexes and plot original graph with quant figure
                t_ind2 = np.array((df2.index - df2.index[0]).total_seconds())
                fig,axe = plt.subplots(4,4,figsize = (12,20),sharex = True)
                for i in range(16):
                    if i+1 not in df2.columns: continue
                    axe[i//4,i%4].plot(t_ind2,df2[i+1])
                
                df_r = filter_erpo(df2)
                replay_data.update({species:df_r.copy()})
                
                for i in range(16):
                    if i+1 not in df2.columns: continue
                    axe[i//4,i%4].plot(t_ind2,df_r[i+1],color = 'red',alpha = 0.75)
                fig.tight_layout()
                
                fig.savefig(r'{}\E_before_after.jpg'.format(r))
                
                #%% Radix
                species = 'R'
                df2 = dfs_copy[species]
                
                #get seconds time indexes and plot original graph with quant figure
                t_ind2 = np.array((df2.index - df2.index[0]).total_seconds())
                fig,axe = plt.subplots(4,4,figsize = (12,20),sharex = True)
                for i in range(16):
                    if i+1 not in df2.columns: continue
                    axe[i//4,i%4].plot(t_ind2,df2[i+1])
                
                df_r = filter_radix(df2)
                replay_data.update({species:df_r.copy()})
                
                #plot amended time series
                for i in range(16):
                    if i+1 not in df2.columns: continue
                    axe[i//4,i%4].plot(t_ind2,df_r[i+1],color = 'red',alpha = 0.75)
                fig.tight_layout()
                
                fig.savefig(r'{}\R_before_after.jpg'.format(r))
                
                #%% write files and zip
                for s in replay_data:
                    filename = '{}_{}'.format(stem.split('-')[0],specie[s])
                    if os.path.isfile(r'{}\{}.csv.zip'.format(root,filename)):
                        print('Files already exist')
                        break
                    compression_options = dict(method='zip', archive_name='{}.csv'.format(filename))
                    replay_data[s].to_csv(r'{}\{}.csv.zip'.format(root,filename))
            except:
                datafailure.append(r)
                continue
            
    
    for f in failed: print('Check starttime for: {}'.format(f))
    for f in nodead: print('Check dead for: {}'.format(f))
    for f in preprocessfailure: print('Check preprocessing for: {}'.format(f))
    for f in datafailure: print('Check dataread for: {}'.format(f))