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
import seaborn as sns
from scipy.stats import linregress as lin


### Parameters global
colors = [
    '#42f5e0','#12aefc','#1612fc','#6a00a3',
    '#8ef743','#3c8f01','#0a4001','#fc03ca',
    '#d9d200','#d96c00','#942c00','#fc2803',
    '#e089b6','#a3a3a3','#7a7a7a','#303030'
    ]

specie = {'E': 'Erpobdella','G':'Gammarus','R':'Radix'}

### functions

"""
Upto date calculation of IGT
"""
def return_IGT_thresh(potable):
    if potable :
        #parameters for potable + quantile 10%
        seuil_bdf = {'G':[0.7,19],'E':[0.7,18],'R':[0.8,5]}
        cutoff = {'G':[1500,2250,12000],'E':[1000,2000,10000],'R':[250,325,1200]}
        palier = {
                'E':np.array([cutoff['E'][1],cutoff['E'][1]*2,cutoff['E'][1]*3,cutoff['E'][2]*3],dtype = int),
                'G':np.array([cutoff['G'][1],cutoff['G'][1]*2,cutoff['G'][1]*3,cutoff['G'][2]*3],dtype = int),
                'R':np.array([cutoff['R'][1],cutoff['R'][1]*2,cutoff['R'][1]*3,cutoff['R'][2]*3],dtype = int)}
        offsets = find_optimum_offsets(palier)
        quant = 0.1
    else :
        #parameters for industriel + quantile 5%
        seuil_bdf = {'G':[0.7,19],'E':[0.7,18],'R':[0.8,5]}
        cutoff = {'G':[2000,3500,12000],'E':[1000,2500,10000],'R':[250,450,1200]}
        palier = {
                'E':np.array([cutoff['E'][1],cutoff['E'][1]*2,cutoff['E'][1]*3,cutoff['E'][2]*3],dtype = int),
                'G':np.array([cutoff['G'][1],cutoff['G'][1]*1.5,cutoff['G'][1]*2.25,cutoff['G'][2]*3],dtype = int),
                'R':np.array([cutoff['R'][1],cutoff['R'][1]*1.5,cutoff['R'][1]*2.25,cutoff['R'][2]*3],dtype = int)}
        offsets = find_optimum_offsets(palier)
        quant = 0.05
        
    return [seuil_bdf,cutoff,offsets,quant]

def find_optimum_offsets(palier = {
        'E':np.array([2500,5000,7000,30000]),
        'G':np.array([3500,5000,8000,30000]),
        'R':np.array([450,600,950,3000])}):
    """ 
    Find log offset for linear fit in higher range in percent IGT
    Will default to in function high range fitpoints ('paliers').
    """
    
    offsets = {'E':1,'G':1,'R':1} #initialise
    for sp in offsets:
        paliers = palier[sp]
        r2 = []
        
        test = np.linspace(0,paliers[0]-1,paliers[0])
        for c in test:
            temp = paliers - c
            r2.append(lin(np.array([1,2,3,4]),np.log(temp))[2])
            
        offsets[sp] = np.argmax(r2)
    
    return offsets

def IGT_bdf(values,species,seuil_bdf,overwrite = None):
    """Bruit de Fond"""
    if overwrite:
        seuil = overwrite[species]
    else:
        seuil = seuil_bdf[species]
    # quantile / facteur
    bdf = np.nanquantile(values,seuil[0])/seuil[1]
    if np.isnan(bdf):
        bdf = 0
    return bdf

def IGT_base(IGT_,species,cutoff,offsets,cut = None):
    """ Find IGT percentage value from percentile IGT """    
    
    if cut:
        seuil = cut[species]
    else:
        seuil  = cutoff[species]
    
    offset = offsets[species]
    if IGT_ < seuil[0]:
        return (IGT_ / (seuil[0]/45))
    elif IGT_ < seuil[1]:
        return ((IGT_ - seuil[0]) / ((seuil[1] - seuil[0])/25)) + 45
    else:
        return 70 + 20 * (np.log((IGT_ - offset)/(seuil[1] - offset))/np.log((seuil[2] - offset)/(seuil[1]-offset)))

def IGT_(values,species,thresh,cut = None,overwrite = None):
    
    seuil_bdf = thresh[0]
    cutoff = thresh[1]
    offsets = thresh[2]
    quant = thresh[3]
    
    IGT_ = np.nanquantile(values,quant)**2
    v = IGT_bdf(values,species,seuil_bdf,overwrite) + IGT_base(IGT_,species,cutoff,offsets,cut) #max()
    if v > 100: v = 100
    return v

"""
Upto date calculation of IGT
"""

"""
Moyenne sur 2 mns
"""
def group_mean(values,dataindex,m,cols,timebin_min = 2):
    # add function for two minute means (mortality should be calculated before this !)
    """
    Create new df with mortality and timestep columns
    Return np array of values
    """
    temp = pd.DataFrame()
    temp = pd.DataFrame(values,index = dataindex, columns = cols)
    temp['Mortality'] = pd.Series(m,index = dataindex)
    temp['timestep'] = timebin_min*((temp.index - temp.index[0]).total_seconds()//(60*timebin_min))
    temp = temp.groupby('timestep').mean()
    temp.index = temp.index.astype(int)
    temp.index = dataindex[0] + pd.to_timedelta(temp.index, unit = 'm')
    values = np.array(temp.drop('Mortality',axis = 1))
    m = np.array(temp['Mortality'])
    return values,m,temp.index

def group_meandf(df,m,timebin_min = 2):
    # add function for two minute means (mortality should be calculated before this !)
    """
    Create new df with mortality and timestep columns
    Return np array of values
    """
    timestep = timebin_min * 60
    
    start_time = df.index[0]
    df['timestep'] = timebin_min*(((df.index - start_time).total_seconds() // timestep))
    df['mortality'] = m
    df_m = df.groupby(['timestep']).mean()
    df_m.index = df_m.index.astype(int)
    df_m.index = df.index[0] + pd.to_timedelta(df_m.index, unit = 'm')
    m = np.array(df_m['mortality'])
        
    return df_m.drop(['mortality'],axis = 1),m

def sort_filedates(files):
    
    """
    Function to return filenames in date order
    Input format : 'toxmate_11122021_13122021.csv'
    """
    
    starts,ends = [],[]
    for file in files:
        if len(file.split('.')[0].split('_')) == 3:
            start = file.split('.')[0].split('_')[1]
            end = file.split('.')[0].split('_')[2]
            if len(start) < 8:
                start = start[:4] + '2021'
                end = end[:4] + '2021'
            starts.append(start)
            ends.append(end)
        else:
            print('File format, incorrect. ERROR - rename file to format: \n .._ddmmYY_ddmmYY.csv')
            return None
        
    starts = sorted([pd.to_datetime(start,format = ('%d%m%Y')) for start in starts])
    ends = sorted([pd.to_datetime(end,format = ('%d%m%Y')) for end in ends])
    
    #check if start filename (only use day and mont ddmm) and add in this order
    filenames = []
    for x in range(len(starts)):
        for i in range(len(files)):
            if (starts[x].strftime('%d%m%Y')[:4] in files[i]) & (ends[x].strftime('%d%m%Y')[:4] in files[i]): filenames.append(files[i])
        
    return filenames

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
        return [pd.concat(dfs).reset_index()]
    else:
        return dfs


def apply_threshold(dfs,thresholds,distplot = False):
    """
    eliminate false values from the dataframes
    """
    
    #as per specie - delete extreme values
    for specie in dfs:
        
        #plot values
        if distplot: 
            fig = plt.figure(figsize = (14,8))
            axe = fig.add_axes([0.1,0.1,0.8,0.8])
            sns.histplot(np.array(dfs[specie]),bins = 500,ax = axe)
            axe.set_xlim((0,3*thresholds[specie]))
            axe.axvline(thresholds[specie],color = 'r',linestyle = '--')
            axe.set_title('Distribution avant trie - {}'.format(specie))
            
        
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

def remove_org(dfs,thresh_life = {'G':12,'E':12,'R':12},
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
    Calculate in blocks of n minutes (not rolling as with lab data)
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


def read_data_terrain(files,merge,plot = True,timestep = 2,startdate = None,distplot = False,
                      thresholds = {'G':190,'E':180,'R':50}):
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
    
    dfs = read_all_terrain_files(files,merge)
    df = dfs[0]
    
    if startdate:
        df = df[df['time'] > startdate]
    
    df['animal'] = df['replica']
    
    df['dist'] = df['inadist'] + df['smldist'] + df['lardist']
    
    dfs_spec = {}
    
    #print(df.columns)
    dfs_spec.update({'G': df[df['specie'] == 'Gammarus']})
    dfs_spec.update({'E': df[df['specie'] == 'Erpobdella']})
    dfs_spec.update({'R': df[df['specie'] == 'Radix']})
    
    dfs_spec = df_distance(dfs_spec)
    if thresholds:
        dfs_spec = apply_threshold(dfs_spec,thresholds,distplot = distplot)
        print('applying thresholds')
    
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
    
    
def plot_16(df,title = None):
    
    """
    Plot a 16 square subplots
    """
    fig,axe = plt.subplots(4,4,sharex = True, figsize = (20,12))
    for i in df.columns:
        axe[(i-1)//4,(i-1)%4].plot(df.index,df[i],color = colors[2])
        axe[(i-1)//4,(i-1)%4].tick_params(axis='x', rotation=90)
    
    if title: plt.suptitle(title)    
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
    threshold_death = {'E':180*3,'G':60*3,'R':360*3}
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

"""
OLD
"""

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

def gen_txt(res,file,species,output = 'Suez',tox = 'TOF771'):
    
    databases = {
        'TOF771':'f5683285-b5fa-4be0-be99-6e20a112fad5'}
    
    root = os.getcwd()
    os.chdir(r'D:\VP\Viewpoint_data\DatabaseFiles\{}'.format(output))
    
    specs = {'G':'Gammarus','E':'Erpobdella','R':'Radix'}
    spec = specs[species]
        
    filename = file.split('.')[0] + species + '.txt'
    
    #unix timestamp
    res['time'] = res.index.astype(np.int64)
    
    with open(filename, 'w', newline = '\n') as f:
        
        #add header
        for i in range(res.shape[0]):
            f.write('{},sensor={} mortality={},toxicityindex={} {}\n'.format(tox,spec,res.iloc[i]['Mortality'],res.iloc[i]['IGT'],int(res.iloc[i]['time'])))
    
    os.chdir(root)

def double_vertical_plot(set1,set2,ind = [],vert = 2,extrasets = None):
    """
    Plot timeline plots one above other default to 2 sets, possible to add using extras function

    Parameters
    ----------
    set1 : 1D np.arr
        first time series.
    set2 : 1D np.arr
        second time series.
    ind : list, optional
        Len of time series for x values. The default is None.
    vert : int, optional
        number of vert stacks
    extrasets : list of np.arr, optional
        vert - 2 should math len on of extrasets
        In each list entry is another timeseries in 1D timeseries

    Returns
    -------
    plot items.

    """
    if not any(ind): ind = set1.index
    fig,axe = plt.subplots(vert,1,figsize = (18,9),sharex = True)
    plt.suptitle('IGT 5% vs. percent new_IGT')
    axe[0].plot(ind,set1,color = 'green')
    axe[1].plot(ind,set2,color = 'green')
    end = 1
    if vert > 2:
        extras = vert - 2
        for i in range(extras):
            axe[2+i].plot(ind,extrasets[i],color = 'green')
        end = 2+i
    
    
    axe[end].tick_params(axis='x', rotation=90)
    return fig,axe

def IGT_percent_and_old(val,species,emptyarr1,emptyarr2,thresh):
    """
    Calculate both new method with percentage and old IGT

    Parameters
    ----------
    val : np.array 1D
        values to be used for calculations.

    Returns
    -------
    Two full np.arrays.

    """
    
    IGTper = emptyarr1
    old_IGT = emptyarr2
    for i in range(len(val)):
        
        IGTper[i] = IGT_(val[i],species,thresh)
        
        #check 100% mortality
        if np.isnan(val[i][0]):
            old_IGT[i] = 0
        else:
            old_IGT[i] = np.quantile(val[i][~np.isnan(val[i])],thresh[-1])**2
    
    return IGTper,old_IGT

def add_mortality(fig,axe,m,ind = []):
    axe_2 = [ax.twinx() for ax in axe]
    for ax in axe_2: ax.set_ylim(top = 1)
    for ax in axe_2:
        ax.plot(ind,m,'orange',linestyle = (0,(1,10)))
        ax.fill_between(ind,m,alpha = 0.3,color = 'orange')
    return fig,axe,axe_2