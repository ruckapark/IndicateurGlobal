# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 16:44:49 2021

Read terrain data

@author: Admin
"""

import os
import pandas as pd
import numpy as np
import shutil
import matplotlib.pyplot as plt
#import seaborn as sns
#from datetime import timedelta
#from data_merge import merge_dfs

#### General parameters (could be made into class)

colors = [
    '#42f5e0','#12aefc','#1612fc','#6a00a3',
    '#8ef743','#3c8f01','#0a4001','#fc03ca',
    '#d9d200','#d96c00','#942c00','#fc2803',
    '#e089b6','#a3a3a3','#7a7a7a','#303030'
    ]

specie = {'E': 'Erpobdella','G':'Gammarus','R':'Radix'}
species = 'G'

thresholds = {'G':190,'E':180,'R':5000}


### functions

def join_text(directory,filename = 'output.txt'):
    
    """
    When generating txt files to load into influx db
    
    Use join text to merge all results from a directory into a text file
    """
    header = r'D:\VP\Viewpoint_data\code\header.txt'
    
    os.chdir(r'{}'.format(directory))
    
    with open(filename,'wb') as wfd:
        files = [header]
        files.extend([i for i in os.listdir() if '.txt' in i])
        for f in files:
            with open(f,'rb') as fd:
                shutil.copyfileobj(fd, wfd)


def read_all_terrain_files(files,merge = False):
    """
    Similar to reading lab files this will:
        
        - read all files in the directory into a dataframe
        - print info about them
        - delete head and tail unecessary
    """
    
    dfs = []                    
    
    for file in files:
        df = pd.read_csv(file,sep = '\t')
        
        print('File: {}\n'.format(file),
              '\tE:{} items\n'.format(len(df[df['specie'] == 'Erpobdella'])),
              '\tR:{} items\n'.format(len(df[df['specie'] == 'Radix'])),
              '\tG:{} items\n'.format(len(df[df['specie'] == 'Gammarus'])))
        
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values(by = ['time','replica'])
        
        df = df.reset_index(drop = True)
        
        print('Before adjustment: total rows{}'.format(len(df)))
        
        """
        check if end includes 16 values (full set)
        if less than 16 - delete end values
        assume this needs to be done once
        repeat for start values
        """
        
        end_time = df.iloc[-1]['time']
        end_data = df[df['time'] == end_time]
        if len(end_data) % 16:
            df = df.iloc[:-len(end_data)]
        
        start_time = df.iloc[0]['time']
        start_data = df[df['time'] == start_time]
        if len(start_data) % 16:
            df = df.iloc[len(start_data):]
            
        start_index = 0
        time_intervals = df.iloc[:48]['time'].unique()
        if len(time_intervals) > 1:
            diffs = []
            for i in range(3):
                diffs.append(df.iloc[16*(i+1)]['time'].second - df.iloc[16*(i)]['time'].second)
                
            diff_dic = dict(zip(diffs,[0,1,2]))
            start_index = 16*diff_dic[max(diffs)]
            
        if start_index:
            df = df.iloc[start_index:]
        
        
        print('After adjustment: total rows{}'.format(len(df)))
        
        dfs.append(df)
    
    #we may want to merge the dfs into one bigger one, or not...
    if merge:
        return None
    else:
        return dfs


def apply_threshold(dfs,thresholds):
    """
    eliminate false values from the dataframes
    """
    
    #as per specie - delete extreme values
    for specie in dfs:
        
        print('Filtering {}: {}/{}values'.format(specie,np.sum(dfs[specie] > thresholds[specie]).sum() , np.size(dfs[specie])))
        dfs[specie][dfs[specie] > thresholds[specie]] = 0
        
    return dfs


def df_distance(dfs):
    
    """
    Transform dfs in dictionary into distance dfs only
    """
    
    for specie in dfs:
        df = dfs[specie]
        timestamps = df['time'].unique()
        animals = list(range(1,17))   
        df_dist = pd.DataFrame(index = timestamps)

        # values assumes that there is the perfect amount
        # there should be a way of matching values (concatenate)
        
        for i in animals:
            temp_df = df[df['animal'] == i][['time','dist']]
            temp_df = temp_df.set_index('time')
            temp_df.index.name = None
            temp_df.columns = [i]
            df_dist = df_dist.join(temp_df)
            
        dfs[specie] = df_dist
        
    return dfs

def remove_org(dfs,thresh_life = {'G':12,'E':12,'R':24},
               thresh_dead = {'G':180,'E':540,'R':360},
               remove = True):
    """
    Assuming remove is true, we just get rid of any organisms with na values above threshold
    """
    for species in dfs:
        df = dfs[species]
        data = np.array(df)
        data_alive = np.ones_like(data)
        data_counters = np.zeros_like(data)
        counters = np.zeros(16)
        alive = np.ones(16)
        
        # through full dataset
        for i in range(thresh_life[species], len(data)):
            
            # through 16
            for x in range(len(data[0])):
                
                # if they are alive
                if alive[x]:
                    
                    if data[i][x]:
                        
                        if data[i-1][x]:
                            counters[x] = 0
                        else:
                            counters[x] += 1
                        
                    else:
                        counters[x] += 1
                
                #if they are dead        
                else:
                    if 0 not in data[(i- thresh_life[species]):i,x]:
                        alive[x] = 1
                        counters[x] = 0
                        
                if counters[x] >= thresh_dead[species]:
                    alive[x] = 0
                    
                data_alive[i] = alive
                data_counters[i] = counters
                
        #replace data with np.nan if dead - use isnull()
        df_alive = pd.DataFrame(data_alive, index = df.index, columns = df.columns)
        df_alive[df_alive == 0] = np.nan
        deathcount = df_alive.isnull().sum()
        
        #check which have more than 20%
        deathcount = deathcount > (len(data)/5)
        
        #retain False columns
        df = df[deathcount[~deathcount].index]
        dfs[species] = df
        print('Idividus: {} are not included in the {} study'.format(deathcount[deathcount].index,specie[species]))
        
        
    return dfs

def df_movingmean(dfs,timestep):
    
    """
    Calculate in blocks of 10 minutes (not rolling as with lab data)
    """
    timestep = timestep * 60
    dfs_mean = {}
    
    for specie in dfs:
        df = dfs[specie].copy()
        start_time = df.index[0]
        df['timestep'] = ((df.index - start_time).total_seconds() // timestep).astype(int)
        
        df_mean = df.groupby(['timestep']).mean()
        index_map = dict(zip(df['timestep'].unique(),[df[df['timestep'] == i].index[-1] for i in df['timestep'].unique()]))
        df_mean = df_mean.set_index(df_mean.index.map(index_map))
        
        dfs_mean.update({specie:df_mean})
        
    return dfs_mean


def read_data_terrain(files,plot = True,timestep = 10):
    """
    Parameters
    ----------
    dir : directory
        Contains csv files output from ToxMate (NOT VPCore2).
        Probably one aval, amont

    Returns
    -------
    dataframes - coherent to the form of the dataframe in read_data_VPCore2.
    """
    
    dfs = read_all_terrain_files(files)
    df = dfs[0]
    
    df['animal'] = df['replica']
    
    df['dist'] = df['inadist'] + df['smldist'] + df['lardist']
    
    dfs_spec = {}
    
    #print(df.columns)
    dfs_spec.update({'G': df[df['specie'] == 'Gammarus']})
    dfs_spec.update({'E': df[df['specie'] == 'Erpobdella']})
    dfs_spec.update({'R': df[df['specie'] == 'Radix']})
    
    dfs_spec = df_distance(dfs_spec)
    dfs_spec = apply_threshold(dfs_spec,thresholds)
    
    dfs_spec_mean = df_movingmean(dfs_spec,timestep)
    
    print(dfs_spec['G'].columns)
    
    return dfs_spec,dfs_spec_mean


def single_plot16(df,species,title = ''):
    fig = plt.figure(figsize = (13,8))
    axe = fig.add_axes([0.1,0.1,0.8,0.8])
    for i in df.columns:
        axe.plot(df.index,df[i],label = '{}{}'.format(species,i),color = colors[i-1])
    axe.tick_params(axis='x', rotation=90)
    axe.set_title(title)
    plt.legend()
    return fig,axe
    
    
def plot_16(df):
    
    """
    Plot a 16 square subplots
    """
    fig,axe = plt.subplots(4,4,sharex = True, figsize = (20,12))
    for i in df.columns:
        axe[(i-1)//4,(i-1)%4].plot(df.index,df[i],color = colors[2])
        axe[(i-1)//4,(i-1)%4].tick_params(axis='x', rotation=90)
        
    return fig,axe

def single_plot(series,species,title = ''):
    fig = plt.figure(figsize = (13,8))
    axe = fig.add_axes([0.1,0.1,0.8,0.8])
    axe.plot(series.index,series)
    axe.set_title(title)
    axe.tick_params(axis='x', rotation=90)
    return fig,axe


def search_dead(data,species):
    
    """
    Search for death in historical data    
    """
    
    #mins (*3 timestep 20s)
    threshold_death = {'E':180*3,'G':60*3,'R':120*3}
    thresh_life = {'G':4*3,'E':4*3,'R':6*3}
    
    data_alive = np.ones_like(data) 
    data_counters = np.zeros_like(data)
    
    # live storage
    counters = np.zeros(16)
    alive = np.ones(16)
    
    # through full dataset
    for i in range(thresh_life[species], len(data)):
        
        # through 16
        for x in range(len(data[0])):
            
            # if they are alive
            if alive[x]:
                
                if data[i][x]:
                    
                    if data[i-1][x]:
                        counters[x] = 0
                    else:
                        counters[x] += 1
                    
                else:
                    counters[x] += 1
            
            #if they are dead        
            else:
                if 0 not in data[(i- thresh_life[species]):i,x]:
                    alive[x] = 1
                    counters[x] = 0
                    
            if counters[x] >= threshold_death[species]:
                alive[x] = 0
                
            data_alive[i] = alive
            data_counters[i] = counters
            
    return data_alive,data_counters


## replay database functions
    
def sigmoid_coeffs(m,species):
    
    """
    Return sigmoid function coefficients,
    depend on # morts in system
    """
    
    #n = number vivant
    n = int(16*(1-m))
    
    a = {'G':3.5,'E':3.5,'R':6.0}
    b = {'G':3.0,'E':2.5,'R':2.5}
    
    if n <= 1: 
        return np.array([0])
    elif (n > 1) & (n < 7):
        x = np.arange(2,n+2)
        return 2* (-1/(1+a[species]**(-x+b[species])) + 1) + 0.15
    else:
        x = np.arange(1,n+1)
        return 2* (-1/(1+a[species]**(-x+b[species])) + 1) + 0.15
    
def percent_IGT(data,species):
    
    """
    scale from sigmoid RAW -> %
    """
    # 0-10 %
    data[data <= 40] = data[data <= 40] /4
    # scale 10+ %
    data[data > 40] = (np.log10(data[data > 40] - 30))*22 + 9
    
    #moving mean
    data = np.array(pd.Series(data).rolling(10).mean().fillna(0))
    
    return data

def IGT_per(df,alive,m,species):
    """
    Combine functions simgoid and percentage to one function
    """
    data = np.array(df)
    data[alive == 0] ==  np.nan
    data.sort()
    output = np.zeros_like(m)
    for i in range(len(output)):
        coeffs = sigmoid_coeffs(m[i],species)
        output[i] = np.sum(data[i][:len(coeffs)]**coeffs)
    return percent_IGT(output,species)
    
def save_results(ind,IGT,m,species,filename):
    
    #make df to write values to .txt file for database
    res = pd.DataFrame(columns = ['IGT','Mortality'],index = ind)
    res['IGT'] = IGT
    res['Mortality'] = m
    return res

def gen_txt(res,file,species,output = 'Suez_res'):
    
    root = os.getcwd()
    os.chdir(r'D:\VP\Viewpoint_data\{}'.format(output))
    
    specs = {'G':'Gammarus','E':'Erpobdella','R':'Radix'}
    spec = specs[species]
    
    os.chdir(r'D:\VP\Viewpoint_data\Suez_res')
        
    filename = file.split('.')[0] + species + '.txt'
    
    res['time'] = res.index.astype(np.int64)
    
    with open(filename, 'w', newline = '\n') as f:
        
        for i in range(res.shape[0]):
            f.write('f5683285-b5fa-4be0-be99-6e20a112fad5,sensor={} mortality={},toxicityindex={} {}\n'.format(spec,res.iloc[i]['Mortality'],res.iloc[i]['IGT'],int(res.iloc[i]['time'])))
    
    os.chdir(root)


if __name__ == '__main__':

    root = r'D:\VP\Viewpoint_data\Suez'
    os.chdir(root)
    files = [f for f in os.listdir() if '2903.csv' in f]
    
    print('The following files will be studied:')
    print(files)
    
    dfs,dfs_mean = read_data_terrain(files)
    
    
    timestep = 10
    
    species = 'R'
    df = dfs[species]
    df_mean = dfs_mean[species]
    
    
    # plot all on same figure - no mean and mean
    single_plot(df,species,title = 'Distance covered')
    single_plot(df_mean,species,title = 'Distance covered movingmean')
    
    # plot individually (plot16)
    fig,axe = plot_16(df)
    fig,axe = plot_16(df_mean)    
    
    #plot IGT with mortality
    data_alive,data_counters = search_dead(np.array(df),species)
    m = np.ones(len(data_alive),dtype = float) - (np.sum(data_alive,axis = 1))/16
    
    values = np.array(df)
    values[data_alive == 0] = np.nan
    # values[i][0] < values[i][1] < values[i][2]
    values.sort()
    
    IGT = np.zeros_like(m)
    old_IGT = np.zeros_like(m)
    for i in range(len(values)):
        coeffs = sigmoid_coeffs(m[i],species)
        IGT[i] = np.sum(values[i][:len(coeffs)]**coeffs)
        #check if all values nan (100% mortality)
        if np.isnan(values[i][0]):
            old_IGT[i] = 0
        else:
            old_IGT[i] = np.quantile(values[i][~np.isnan(values[i])],0.05)**2
       
    # caluclate IGT from raw -> %    
    IGT = percent_IGT(IGT, species)
    
    #compare old and new values
    fig,axe = plt.subplots(2,1,figsize = (18,9),sharex = True)
    plt.suptitle('IGT 5% vs. percent new_IGT')
    axe[0].plot(df.index,old_IGT,color = 'green')
    axe[1].plot(df.index,IGT,color = 'green')
    axe[1].tick_params(axis='x', rotation=90)