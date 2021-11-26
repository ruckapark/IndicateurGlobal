# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 13:33:37 2021

Temoin tests 

@author: Admin
"""

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
from dope_reg import dope_read
import dataread as d_
os.chdir('..')

plt.close('all')

def read_merge(files):
    """
    Merge all the data in the list files
    
    At a later point this could be made into the same function as terrain
    """
    
    print('The following files will be merged:')
    print(files)
    
    dfs = []
    for file in files:
        df = pd.read_csv(file,sep = '\t',encoding = 'utf-16')    #read each df in directory df
        df = df[df['datatype'] == 'Locomotion']                         #store only locomotion information
    
        #Error VPCore2
        #conc,subs = df['Conc'].iloc[0],df['Sub'].iloc[0]
    
        #sort values sn = , pn = ,location = E01-16 etcc., aname = A01-04,B01-04 etc.
        df = df.sort_values(by = ['sn','pn','location','aname'])
        df = df.reset_index(drop = True)
    
        #treat time variable - this gets the days and months the wrong way round
        df['time'] = pd.to_datetime(df['stdate'] + " " + df['sttime'], format = '%d/%m/%Y %H:%M:%S')
        
        maxrows = len(df)//48
        print('Before adjustment: total rows{}'.format(len(df)))
        df = df.iloc[:maxrows*48]
        print('After adjustment: total rows{}'.format(len(df)))
        dfs.append(df)
        
    return merge_dfs(dfs)


def preproc(df):
    """
    Preprocessing of the df to get it in correct form
    Return dictionary of dfs for each species - only with distances
    """
    
    #column for specie
    mapping = lambda a : {'E': 'Erpobdella','G':'Gammarus','R':'Radix'}[a[0]]
    df['specie'] = df['location'].map(mapping)
    
    #moi le column 'animal' n'a que des NaNs
    good_cols = ['time','location','stdate','specie','inact','inadur','inadist','smlct','smldur','smldist','larct','lardur','lardist','emptyct','emptydur']
    df = df[good_cols]
    
    #create animal 1-16 for all species
    df['animal'] = df['location'].str[1:].astype(int)
    
    df['abtime'] = df['time'].astype('int64')//1e9 #convert nano
    df['abtime'] = df['abtime'] - df['abtime'][0]
    
    #total distance inadist is only zeros?
    df['dist'] = df['inadist'] + df['smldist'] + df['lardist']
    
    #seperate into three dfs
    dfs = {}
    for spec in specie:
        
        temp = df[df['specie'] == specie[spec]]   
        timestamps = temp['time'].unique()
        animals = temp['animal'].unique()   
        df_dist = 0
        df_dist = pd.DataFrame(index = timestamps,columns = animals)
        
        for i in animals:
            temp_df = temp[temp['animal'] == i]
            df_dist[i] = temp_df['dist'].values
        
        dfs.update({spec:df_dist})
        
    return dfs

def remove_dead_test(df):
    
    """
    - Taken idea from main but simpler for smaller files 
    - If percentage of 0s is high, remove individual from test
    
    """
    
    dead = []
    for col in df.columns:
        if df[col].value_counts()[0]/df[col].shape[0] > 0.98: dead.append(col)
    print('Remove : ',dead)
    return dead

def rolling_mean(df,timestep):
    
    #convert mins to number of 20 second intervals
    timestep = (timestep * 60)//20
    return df.rolling(timestep).mean().dropna()

def single_plot(series,title = ''):
    fig = plt.figure(figsize = (13,8))
    axe = fig.add_axes([0.1,0.1,0.8,0.8])
    axe.plot(series.index,series)
    axe.set_title(title)
    axe.tick_params(axis = 'x', rotation = 90)
        
    return fig,axe

def plot_16(df,date_range,title = None):
    
    """
    Plot a 16 square subplots
    """
    fig,axe = plt.subplots(4,4,sharex = True,sharey = True,figsize = (20,12))
    for i in df.columns:
        axe[(i-1)//4,(i-1)%4].plot(df.index,df[i],color = colors[2])
        axe[(i-1)//4,(i-1)%4].axvspan(date_range[0], date_range[1], alpha=0.7, color='orange')
        axe[(i-1)//4,(i-1)%4].axvline(date_range[0] + pd.Timedelta(seconds = 30), color = 'red')
        axe[(i-1)//4,(i-1)%4].tick_params(axis='x', rotation=90)
        
    if title:
        plt.suptitle(title)

def dataplot_mark_dopage(axe,date_range):
    """
    Shade the doping period and mark the doping moment thoroughly
    """
    #shade over doping period - item extracts value from pandas series
    axe.axvspan(date_range[0], date_range[1], alpha=0.7, color='orange')
    axe.axvline(date_range[0] + pd.Timedelta(seconds = 30), color = 'red')

#take strategy from weekly results

if __name__ == '__main__':
    
    # parametres a modifier
    Tox = 765
    species = 'G'
    etude_ = 1
    specie = {'E': 'Erpobdella','G':'Gammarus','R':'Radix'}
    
    #find directory
    os.chdir(r'D:\VP\Viewpoint_data\Temoins\Tem_{}_0{}'.format(Tox,etude_))
    files = [file for file in os.listdir() if os.path.isfile(file)]
    
    #read files and wrangle
    df = d_.read_merge(files)
    dfs = d_.preproc(df)
    
    #select gammarus df
    df = dfs[species]
    df = df.drop(columns = d_.remove_dead(df))
    
    temoin_df = dope_read(reg = 'temoin_reg')
    
    temoin = temoin_df[temoin_df['TxM'] ==  Tox].iloc[etude_ - 1]
    case = 'Tox:{},  Methode: {}, Volume: {}, Date: {} {}-{}'.format(
        Tox,
        temoin['Molecule'],
        temoin['Concentration'],
        temoin['Start'].strftime("%m-%d"),
        temoin['Start'].strftime("%H:%M:%S"),
        temoin['End'].strftime("%H:%M:%S"))
    
    df_pre = df[df.index < temoin['Start']]
    df_post = df[(df.index > temoin['End']) & (df.index < (temoin['End']+pd.Timedelta(minutes = 30)))]
    
    t_mins = 5
    df_mean = d_.rolling_mean(df,t_mins)
    
    #plot for 1 hour scale
    if df.index[-1] > (temoin['End'] + pd.Timedelta(hours = 1)):
        d_.plot_16(df_mean[df_mean.index < (temoin['End']+pd.Timedelta(hours = 1))],[temoin['Start'],temoin['End']],title = 'Distance/20s -  {}'.format(case))
    else:
        d_.plot_16(df_mean,[temoin['Start'],temoin['End']],title = 'Distance/20s -  {}'.format(case))  
    
    #plot for 15 minutes
    if temoin['Molecule'] != np.nan:
        short = df_mean[(df_mean.index > (temoin['Start']-pd.Timedelta(minutes = 6))) & ((df_mean.index < temoin['Start']+pd.Timedelta(minutes = 17)))]
        d_.plot_16(short,[temoin['Start'],temoin['End']],title = case)
    
    
    #plot of means axis = 1
    means = df_mean.median(axis = 1)
    fig,axe = d_.single_plot(means,title = 'Mean -  {}'.format(case))
    d_.dataplot_mark_dopage(axe,[temoin['Start'],temoin['End']])