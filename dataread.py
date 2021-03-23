# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 10:41:44 2020

Test file for the TxM 765

@author: Admin
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
from data_merge import merge_dfs


# Covert date for corrupt VPCore2 files

def convert_date(filename):
    """
    format 20201230-093312.xls becomes dt(dd30/mm12/YYYY2020) and dt(09:33:12) 
    """
    day_unformatted = filename.split('-')[0]
    day_formatted = day_unformatted[:4] + '/' + day_unformatted[4:6] + '/' + day_unformatted[6:]
    
    if len(filename.split('-')) > 2:
        hour_unformatted = filename.split('-')[1]    
    else:
        hour_unformatted = filename.split('-')[1].split('.')[0]
    hour_formatted = hour_unformatted[:2] + ':' + hour_unformatted[2:4] + ':' + hour_unformatted[4:]
        
    date = day_formatted + ' ' + hour_formatted
    
    return pd.to_datetime(date, format = "%Y/%m/%d %H:%M:%S")
def correct_dates(file):    

    #extract date from target_file (it is a .xls but should be read as csv)
    true_date = convert_date(file.split('\\')[-1])
    
    #read in data
    df = pd.read_csv(file,sep = '\t',encoding = 'utf-16')
    
    #make new np vector/array of all the combines datetimes (lambda function)
    df['time'] = pd.to_datetime(df['stdate'] + " " + df['sttime'], format = '%d/%m/%Y %H:%M:%S')
    
    #redo dates
    false_date = df['time'].min()
    if false_date < true_date:
        diff = true_date - false_date
        df['time'] = df['time'] + diff
    else:
        diff = false_date - true_date
        df['time'] = df['time'] - diff
    
    # from time column, rewrite 'stdate' and 'sttime'
    df['stdate'] = df['time'].dt.strftime('%d/%m/%Y')
    df['sttime'] = df['time'].dt.strftime('%H:%M:%S')
        
    # delete time column
    df = df.drop('time',1)
    
    #make new_file (add copy at end without deleting original first)
    df.to_csv(file.split('.')[0] + '-copy.xls', sep = '\t', encoding = 'utf-16')
    
# 


def main(Tox,species):
    
    """
    Directory must only include files from one experiment.
    """
    
    specie = {'E': 'Erpobdella','G':'Gammarus','R':'Radix'}

    #os.chdir(r'D:\VP\Viewpoint_data\TxM{}-PC'.format(Tox))
    os.chdir(r'D:\VP\Viewpoint_data\TxM{}-PC'.format(Tox))
    files = os.listdir()
    
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
        
    df = merge_dfs(dfs)
        
    #E01 etc.
    mapping = lambda a : {'E': 'Erpobdella','G':'Gammarus','R':'Radix'}[a[0]]
    df['specie'] = df['location'].map(mapping)
    
    #moi le column 'animal' n'a que des NaNs
    good_cols = ['time','location','stdate','specie','entct','inact','inadur','inadist','smlct','smldur','smldist','larct','lardur','lardist','emptyct','emptydur']
    df = df[good_cols]
    
    #create animal column from location E01
    df['animal'] = df['location'].str[1:].astype(int)
    
    df['dose'] = '72ug/L'
    df['etude'] = 'ETUDE001'
    df['lot'] = 'Zn'
    
    isnull = df.isnull().sum() #can check var explorer
    plt.figure()
    sns.heatmap(df.isnull(), yticklabels = False, cbar = False)
    
    #sml distance all nans? replace with zeros
    df['smldist'] = df['smldist'].fillna(0)
    
    # doesn't appear necessary - what is it?
    df = df.drop('entct',axis = 1)
    
    # add channel column (doesn't seem to be used in IGT script)
    df['channel'] = 19139235
    
    #%%
    """
    Datetime is in nanoseconds, //10e9 to get seconds
    Can zero this value
    """
    #recreate abtime as seconds since first value - For some reason creates a huge diff day to day
    df['abtime'] = df['time'].astype('int64')//1e9 #convert nano
    df['abtime'] = df['abtime'] - df['abtime'][0]
    
    #create threshold columns
    df['threshold'] = 10
    df['thresholdHigh'] = 20
    df['thresholdLow'] = 5
    df['protocole'] = 1
    
    
    
    #species is input
    df = df[df['specie'] == specie[species]]
    
    #total distance inadist is only zeros?
    df['dist'] = df['inadist'] + df['smldist'] + df['lardist']
    
    #plot all animals
    fig = plt.figure()
    axe = fig.add_axes([0.1,0.1,0.8,0.8])
    for i in range(16):
        axe.plot(df[df['animal']==(i+1)]['time'],df[df['animal']==(i+1)]['dist'],label = 'gamm{}'.format(i+1))
    axe.set_title('Mean values for Gammare')
    
    
    # this does not work
    timestep = 1*60 # 1 minute
    df['timestep'] = df['abtime']//timestep * timestep
    
    
    timesteps = df['timestep'].unique().astype(int)
    animals = range(1,17)
    
    #append mean distance 1 by 1
    df_mean_dist = pd.DataFrame(index = timesteps)
    
    #groupby animal method? - create df with column as cell and mean distances
    for i in animals:
        temp_df = df[df['animal'] == i]
        mean_distance = temp_df.groupby(['timestep']).mean()['dist']
        df_mean_dist[i] = mean_distance
        
    #plot all the signals averaged out
    fig_mean_tstep = plt.figure()
    axe_mean_tstep = fig_mean_tstep.add_axes([0.1,0.1,0.8,0.8])
    for i in animals:
        axe_mean_tstep.plot(df_mean_dist.index,df_mean_dist[i],label = 'gamm {}'.format(i))
    fig_mean_tstep.show()
    
    #plot all the means across cells
    mean_dist = df_mean_dist.mean(axis = 1)
    
    fig_mean = plt.figure()
    axe_mean = fig_mean.add_axes([0.1,0.1,0.8,0.8])
    axe_mean.plot(mean_dist.index,mean_dist)
    fig_mean.show()
    
    #plot quantile 0.05 across cells
    quantile_dist = df_mean_dist.quantile(q = 0.05, axis = 1)**2
    fig_quant = plt.figure()
    axe_quant = fig_quant.add_axes([0.1,0.1,0.8,0.8])
    axe_quant.plot(quantile_dist.index,quantile_dist)
    fig_quant.show()