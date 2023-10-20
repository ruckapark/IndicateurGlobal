# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 11:22:47 2023

Use the distribution of the preceding time, if possible to evaluate a baseline for the high activity

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

def get_IGT(qlow,qhigh):
    
    qlow,qhigh = qlow.copy(),qhigh.copy()
    qlow[qlow < 0.2] = 0.0
    qhigh[qhigh > -0.2] = 0.0
    
    IGT = pd.DataFrame(index = qlow.index,columns = ['low','high','total'])
    IGT['low'],IGT['high'] = qlow,qhigh
    IGT['total'] = (1*(IGT['low'] > 0)) * (1*(IGT['high'] < 0))
    
    IGT_array = np.zeros(IGT.shape[0])
    
    for i in range(IGT.shape[0]):
        
        if IGT['total'].iloc[i] == 0:
            if IGT['low'].iloc[i] > 0:
                IGT_array[i] = IGT['low'].iloc[i]
            else:
                IGT_array[i] = IGT['high'].iloc[i]
        else:
            if IGT['low'].iloc[i] > abs(IGT['high'].iloc[i]):
                IGT_array[i] = IGT['low'].iloc[i]
            else:
                IGT_array[i] = IGT['high'].iloc[i]
    
    return pd.Series(IGT_array,index = IGT.index)

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
    
    reference_distributions = {
        'E':{'median':80,'std':20},
        'G':{'median':90,'std':80},
        'R':{'median':6.45,'std':3}
        }
    
    specie = {'E': 'Erpobdella','G':'Gammarus','R':'Radix'}
    dfs = {}
    root = methomyls[0]
    
    dope_df = dope_read_extend()
    
    #IGT values
    optimum = 0.129167
    optimum2 = 0.1909
    
    qs_low = {'E':optimum,'G':optimum,'R':optimum}
    qs_high = {'E':1-optimum,'G':0.95,'R':1-optimum2}
    
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
    plt.close('all')
    
    #1 Smoothed quantiles
    s = 'E'
    df = dfs[s]
    
    #mean treatment of data
    t_mins = 5
    
    #lower quantile
    quantile_distRAW = df.quantile(q = optimum, axis = 1)
    quantile_low = d_.rolling_mean(df.quantile(q = qs_low[s], axis = 1),t_mins)
    quantile_high = d_.rolling_mean(df.quantile(q = qs_high[s], axis = 1),t_mins)
    
    #visualise IGT
    fig = plt.figure(figsize = (13,7))
    axe = fig.add_axes([0.1,0.1,0.8,0.8])
    axe.plot(quantile_low.index,quantile_low,'blue')
    axe.plot(quantile_high.index,quantile_high,'red')
    axe.axvline(dopage,color = 'black')
    
    #pre spike reference data
    pre_spike = df[(df.index < dopage) & (df.index > dopage - pd.Timedelta(hours = 2))]
    quantile_high_pre = d_.rolling_mean(pre_spike.quantile(q = qs_high[s], axis = 1),t_mins)
    
    pre_spike_parameters = {'median':quantile_high_pre.median(),'std':quantile_high_pre.std()}
    range_low,range_high = reference_distributions[s]['median'] - reference_distributions[s]['std'],reference_distributions[s]['median'] + reference_distributions[s]['std']
    
    if range_low < pre_spike_parameters['median'] < range_high:
        print('ok')
        
        axe.axhline(pre_spike_parameters['median'],color = 'blue')
        axe.axhline(pre_spike_parameters['median']-2*pre_spike_parameters['std'],color = 'blue',linestyle = '--')
        axe.axhspan(pre_spike_parameters['median'], pre_spike_parameters['median']-2*pre_spike_parameters['std'], facecolor='orange', alpha=0.5)
        
        quantile_high = quantile_high - (pre_spike_parameters['median']-2*pre_spike_parameters['std'])
        quantile_high[quantile_high > 0] = 0.0
        
        quantile_low = quantile_low**2
        quantile_high = -(quantile_high**2)
        
        fig = plt.figure(figsize = (13,7))
        axe = fig.add_axes([0.1,0.1,0.8,0.8])
        
        axe.plot(quantile_low.index,quantile_low,color = 'blue')
        axe.plot(quantile_high.index,quantile_high,color = 'red')
        axe.axvline(dopage,color = 'black')
        
        IGT = get_IGT(quantile_low,quantile_high)
        axe.plot(IGT.index,IGT.values,color = 'black',alpha = 0.5)
        
    else:
        print('Not stable dataset')
    
    
    s = 'R'
    df = dfs[s]
    
    #mean treatment of data
    t_mins = 5
    
    #lower quantile
    quantile_distRAW = df.quantile(q = optimum, axis = 1)
    quantile_low = d_.rolling_mean(df.quantile(q = qs_low[s], axis = 1),t_mins)
    quantile_high = d_.rolling_mean(df.quantile(q = qs_high[s], axis = 1),t_mins)
    
    #visualise IGT
    fig = plt.figure(figsize = (13,7))
    axe = fig.add_axes([0.1,0.1,0.8,0.8])
    axe.plot(quantile_low.index,quantile_low,'blue')
    axe.plot(quantile_high.index,quantile_high,'red')
    axe.axvline(dopage,color = 'black')
    
    #pre spike reference data
    pre_spike = df[(df.index < dopage) & (df.index > dopage - pd.Timedelta(hours = 2))]
    quantile_high_pre = d_.rolling_mean(pre_spike.quantile(q = qs_high[s], axis = 1),t_mins)
    
    pre_spike_parameters = {'median':quantile_high_pre.median(),'std':quantile_high_pre.std()}
    range_low,range_high = reference_distributions[s]['median'] - reference_distributions[s]['std'],reference_distributions[s]['median'] + reference_distributions[s]['std']
    
    if range_low < pre_spike_parameters['median'] < range_high:
        print('ok')
        
        axe.axhline(pre_spike_parameters['median'],color = 'blue')
        axe.axhline(pre_spike_parameters['median']-2*pre_spike_parameters['std'],color = 'blue',linestyle = '--')
        axe.axhspan(pre_spike_parameters['median'], pre_spike_parameters['median']-2*pre_spike_parameters['std'], facecolor='orange', alpha=0.5)
        
        quantile_high = quantile_high - (pre_spike_parameters['median']-2*pre_spike_parameters['std'])
        quantile_high[quantile_high >= 0] = 0.0
        
        quantile_low = quantile_low**2
        quantile_high = -(quantile_high**2)
        
        fig = plt.figure(figsize = (13,7))
        axe = fig.add_axes([0.1,0.1,0.8,0.8])
        
        axe.plot(quantile_low.index,quantile_low,color = 'blue')
        axe.plot(quantile_high.index,quantile_high,color = 'red')
        axe.axvline(dopage,color = 'black')
        
        IGT = get_IGT(quantile_low,quantile_high)
        axe.plot(IGT.index,IGT.values,color = 'black',alpha = 0.5)
        
    else:
        print('Not stable dataset')
        
    s = 'G'
    df = dfs[s]
    
    #mean treatment of data
    t_mins = 5
    
    #lower quantile
    quantile_distRAW = df.quantile(q = optimum, axis = 1)
    quantile_low = d_.rolling_mean(df.quantile(q = qs_low[s], axis = 1),t_mins)
    quantile_high = d_.rolling_mean(df.quantile(q = qs_high[s], axis = 1),t_mins)
    
    #visualise IGT
    fig = plt.figure(figsize = (13,7))
    axe = fig.add_axes([0.1,0.1,0.8,0.8])
    axe.plot(quantile_low.index,quantile_low,'blue')
    axe.plot(quantile_high.index,quantile_high,'red')
    axe.axvline(dopage,color = 'black')
    
    #pre spike reference data
    pre_spike = df[(df.index < dopage) & (df.index > dopage - pd.Timedelta(hours = 2))]
    quantile_high_pre = d_.rolling_mean(pre_spike.quantile(q = qs_high[s], axis = 1),t_mins)
    
    pre_spike_parameters = {'median':quantile_high_pre.median(),'std':quantile_high_pre.std()}
    range_low,range_high = reference_distributions[s]['median'] - reference_distributions[s]['std'],reference_distributions[s]['median'] + reference_distributions[s]['std']
    
    if range_low < pre_spike_parameters['median'] < range_high:
        print('ok')
        
        axe.axhline(pre_spike_parameters['median'],color = 'blue')
        axe.axhline(pre_spike_parameters['median']-2*pre_spike_parameters['std'],color = 'blue',linestyle = '--')
        axe.axhspan(pre_spike_parameters['median'], pre_spike_parameters['median']-2*pre_spike_parameters['std'], facecolor='orange', alpha=0.5)
        
        quantile_high = quantile_high - (pre_spike_parameters['median']-2*pre_spike_parameters['std'])
        quantile_high[quantile_high >= 0] = 0.0
        
        quantile_low = quantile_low**2
        quantile_high = -(quantile_high**2)
        
        fig = plt.figure(figsize = (13,7))
        axe = fig.add_axes([0.1,0.1,0.8,0.8])
        
        axe.plot(quantile_low.index,quantile_low,color = 'blue')
        axe.plot(quantile_high.index,quantile_high,color = 'red')
        axe.axvline(dopage,color = 'black')
        
        IGT = get_IGT(quantile_low,quantile_high)
        axe.plot(IGT.index,IGT.values,color = 'black',alpha = 0.5)
        
    else:
        print('Not stable dataset')
        
    """
    #%%join datasets to form global IGT signal use methomyls[0]
    quantile_low[quantile_low < 0.5] = 0.0
    quantile_high[quantile_high > -0.5] = 0.0
    
    IGT = pd.DataFrame(index = quantile_low.index,columns = ['low','high','total'])
    IGT['low'],IGT['high'] = quantile_low,quantile_high
    IGT['total'] = (1*(IGT['low'] > 0)) * (1*(IGT['high'] < 0))
    
    IGT_array = np.zeros(IGT.shape[0])
    
    for i in range(IGT.shape[0]):
        
        if IGT['total'].iloc[i] == 0:
            if IGT['low'].iloc[i] > 0:
                IGT_array[i] = IGT['low'].iloc[i]
            else:
                IGT_array[i] = IGT['high'].iloc[i]
        else:
            if IGT['low'].iloc[i] > abs(IGT['high'].iloc[i]):
                IGT_array[i] = IGT['low'].iloc[i]
            else:
                IGT_array[i] = IGT['high'].iloc[i]
                
    axe.plot(IGT.index,IGT_array,'black',alpha = 0.4)
    """
    
    # s = 'E'
    # df = dfs[s].copy()
    
    # #mean treatment of data
    # t_mins = 5
    # df = d_.rolling_mean(df,t_mins)
    
    # #lower quantile
    # quantile_low = df.quantile(q = qs_low[s], axis = 1)
    # quantile_high = df.quantile(q = qs_high[s], axis = 1)
    
    # #visualise IGT
    # fig = plt.figure(figsize = (13,7))
    # axe = fig.add_axes([0.1,0.1,0.8,0.8])
    # axe.plot(quantile_low.index,quantile_low,'blue')
    # axe.plot(quantile_high.index,quantile_high,'red')
    # axe.axvline(dopage,color = 'black')
    
    # #pre spike reference data
    # pre_spike = df[(df.index < dopage) & (df.index > dopage - pd.Timedelta(hours = 2))]
    # quantile_high_pre = d_.rolling_mean(pre_spike.quantile(q = qs_high[s], axis = 1),t_mins)
    
    # pre_spike_parameters = {'median':quantile_high_pre.median(),'std':quantile_high_pre.std()}
    # range_low,range_high = reference_distributions[s]['median'] - reference_distributions[s]['std'],reference_distributions[s]['median'] + reference_distributions[s]['std']
    
    # if range_low < pre_spike_parameters['median'] < range_high:
    #     print('ok')
        
    #     axe.axhline(pre_spike_parameters['median'],color = 'blue')
    #     axe.axhline(pre_spike_parameters['median']-2*pre_spike_parameters['std'],color = 'blue',linestyle = '--')
    #     axe.axhspan(pre_spike_parameters['median'], pre_spike_parameters['median']-2*pre_spike_parameters['std'], facecolor='orange', alpha=0.5)
        
    #     quantile_high = quantile_high - (pre_spike_parameters['median']-2*pre_spike_parameters['std'])
    #     quantile_high[quantile_high > 0] = 0.0
        
    #     quantile_low = quantile_low**2
    #     quantile_high = -(quantile_high**2)
        
    #     fig = plt.figure(figsize = (13,7))
    #     axe = fig.add_axes([0.1,0.1,0.8,0.8])
        
    #     axe.plot(quantile_low.index,quantile_low,color = 'blue')
    #     axe.plot(quantile_high.index,quantile_high,color = 'red')
    #     axe.axvline(dopage,color = 'black')
        
    #     IGT = get_IGT(quantile_low,quantile_high)
    #     axe.plot(IGT.index,IGT.values,color = 'black',alpha = 0.5)
        
    # else:
    #     print('Not stable dataset')
    
    # #2 Unsmoothed quantiles
    # s = 'R'
    # df = dfs[s].copy()
    
    # #mean treatment of data
    # t_mins = 5
    # df = d_.rolling_mean(df,t_mins)
    
    # #lower quantile
    # quantile_low = df.quantile(q = qs_low[s], axis = 1)
    # quantile_high = df.quantile(q = qs_high[s], axis = 1)
    
    # #visualise IGT
    # fig = plt.figure(figsize = (13,7))
    # axe = fig.add_axes([0.1,0.1,0.8,0.8])
    # axe.plot(quantile_low.index,quantile_low,'blue')
    # axe.plot(quantile_high.index,quantile_high,'red')
    # axe.axvline(dopage,color = 'black')
    
    # #pre spike reference data
    # pre_spike = df[(df.index < dopage) & (df.index > dopage - pd.Timedelta(hours = 2))]
    # quantile_high_pre = d_.rolling_mean(pre_spike.quantile(q = qs_high[s], axis = 1),t_mins)
    
    # pre_spike_parameters = {'median':quantile_high_pre.median(),'std':quantile_high_pre.std()}
    # range_low,range_high = reference_distributions[s]['median'] - reference_distributions[s]['std'],reference_distributions[s]['median'] + reference_distributions[s]['std']
    
    # if range_low < pre_spike_parameters['median'] < range_high:
    #     print('ok')
        
    #     axe.axhline(pre_spike_parameters['median'],color = 'blue')
    #     axe.axhline(pre_spike_parameters['median']-2*pre_spike_parameters['std'],color = 'blue',linestyle = '--')
    #     axe.axhspan(pre_spike_parameters['median'], pre_spike_parameters['median']-2*pre_spike_parameters['std'], facecolor='orange', alpha=0.5)
        
    #     quantile_high = quantile_high - (pre_spike_parameters['median']-2*pre_spike_parameters['std'])
    #     quantile_high[quantile_high >= 0] = 0.0
        
    #     quantile_low = quantile_low**2
    #     quantile_high = -(quantile_high**2)
        
    #     fig = plt.figure(figsize = (13,7))
    #     axe = fig.add_axes([0.1,0.1,0.8,0.8])
        
    #     axe.plot(quantile_low.index,quantile_low,color = 'blue')
    #     axe.plot(quantile_high.index,quantile_high,color = 'red')
    #     axe.axvline(dopage,color = 'black')
        
    #     IGT = get_IGT(quantile_low,quantile_high)
    #     axe.plot(IGT.index,IGT.values,color = 'black',alpha = 0.5)
        
    # else:
    #     print('Not stable dataset')