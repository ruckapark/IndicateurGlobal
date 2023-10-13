# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 10:41:44 2020

Test file for the TxM 765

@author: Admin
"""

import os
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from datetime import timedelta,datetime
from data_merge import merge_dfs,merge_dfs_nodatechange

# FUNCTIONS

sns.set_style('darkgrid', {"xtick.major.size": 8, "ytick.major.size": 8})
sns.set_context('paper')
colors = [
    '#42f5e0','#12aefc','#1612fc','#6a00a3',
    '#8ef743','#3c8f01','#0a4001','#fc03ca',
    '#d9d200','#d96c00','#942c00','#fc2803',
    '#e089b6','#a3a3a3','#7a7a7a','#303030'
    ]

def read_merge(files,datechange = True,oldfiles = False):
    """
    Merge all the data in the list files
    
    At a later point this could be made into the same function as terrain
    """
    
    print('The following files will be merged:')
    print(files)
    
    dfs = []
    for file in files:
        if oldfiles :
            df = pd.read_csv(file,sep = '\t')    #read each df in directory df
            
            df = df.sort_values(by = ['utime','specie','condition'])
            df = df.reset_index(drop = True)
            
            df['time'] = pd.to_datetime(df['time'], format = '%Y-%m-%d %H:%M:%S')
            
        
        else :
            df = pd.read_csv(file,sep = '\t',encoding = 'utf-16')
            df = df[df['datatype'] == 'Locomotion']                         #store only locomotion information
    
    
            #sort values sn = , pn = ,location = E01-16 etcc., aname = A01-04,B01-04 etc.
            df = df.sort_values(by = ['sn','pn','location','aname'])
            df = df.reset_index(drop = True)
    
            #treat time variable - this gets the days and months the wrong way round
            try:
                df['time'] = pd.to_datetime(df['stdate'] + " " + df['sttime'], format = '%d/%m/%Y %H:%M:%S')
            except ValueError:
                try:
                    df['time'] = pd.to_datetime(df['stdate'] + " " + df['sttime'], format = '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    df['time'] = pd.to_datetime(df['stdate'] + " " + df['sttime'], format = '%m/%d/%Y %H:%M:%S')
            
        maxrows = len(df)//48
        print('Before adjustment: total rows{}'.format(len(df)))
        df = df.iloc[:maxrows*48]
        print('After adjustment: total rows{}'.format(len(df)))
        dfs.append(df)
        
    if datechange:    
        return merge_dfs(dfs)
    else:
        return merge_dfs_nodatechange(dfs)
    
def read_quant(files,datechange = True):
    
    dfs = []
    for file in files:
        df = pd.read_csv(file,sep = '\t',encoding = 'utf-16')
        df = df[df['datatype'] == 'Quantization'] 

        #sort values sn = , pn = ,location = E01-16 etcc., aname = A01-04,B01-04 etc.
        df = df.sort_values(by = ['sn','pn','location','aname'])
        df = df.reset_index(drop = True)

        #treat time variable - this gets the days and months the wrong way round
        try:
            df['time'] = pd.to_datetime(df['stdate'] + " " + df['sttime'], format = '%d/%m/%Y %H:%M:%S')
        except ValueError:
            try:
                df['time'] = pd.to_datetime(df['stdate'] + " " + df['sttime'], format = '%Y-%m-%d %H:%M:%S')
            except ValueError:
                df['time'] = pd.to_datetime(df['stdate'] + " " + df['sttime'], format = '%m/%d/%Y %H:%M:%S')
        
        maxrows = len(df)//48
        df = df.iloc[:maxrows*48]
        dfs.append(df)
        
    if datechange:    
        return merge_dfs(dfs)
    else:
        return merge_dfs_nodatechange(dfs)
    
    
def preproc(df, oldfiles = False,quant = False):
    """
    Preprocessing of the df to get it in correct form
    Return dictionary of dfs for each species - only with distances
    """
    specie = {'E': 'Erpobdella','G':'Gammarus','R':'Radix'}
    
    if oldfiles :
        df = df.drop(columns = ['animal'])
        df = df.rename(columns = {'specie':'animal'})
        df = df.rename(columns = {'condition':'specie'})
        
        good_cols = ['time','animal','specie','inact','inadur','inadist','smlct','smldur','smldist','larct','lardur','lardist','emptyct','emptydur']
        df = df[good_cols]
    
    else: 
        #column for specie
        mapping = lambda a : {'E': 'Erpobdella','G':'Gammarus','R':'Radix'}[a[0]]
        df['specie'] = df['location'].map(mapping)
        
        #moi le column 'animal' n'a que des NaNs
        good_cols = ['time','location','stdate','specie','inact','inadur','inadist','smlct','smldur','smldist','larct','lardur','lardist','emptyct','emptydur']
        if quant: good_cols = ['time','location','stdate','specie','frect','fredur','midct','middur']
        df = df[good_cols]
        
        #create animal 1-16 for all species
        df['animal'] = df['location'].str[1:].astype(int)
    
    df['abtime'] = df['time'].astype('int64')//1e9 #convert nano
    df['abtime'] = df['abtime'] - df['abtime'][0]
    
    if not quant:
        #total distance inadist is only zeros?
        df['dist'] = df['inadist'] + df['smldist'] + df['lardist']
        
        #seperate into three dfs
        dfs = {}
        for spec in specie:
            
            df_spec = df[df['specie'] == specie[spec]]   
            timestamps = df_spec['time'].unique()
            animals = df_spec['animal'].unique()   
            df_dist = None
            df_dist = pd.DataFrame(index = timestamps,columns = animals)
            
            for i in animals:
                temp_df = df_spec[df_spec['animal'] == i]
                df_dist[i].loc[temp_df['time']] = temp_df['dist'].values
            
            dfs.update({spec:df_dist.fillna(0)})
    else:
        
        
        #total distance inadist is only zeros?
        df['midtime'] = df['middur'] / (df['fredur'] + df['middur'])
        
        #replace nans with 1
        df = df.fillna(1.)
        
        #seperate into three dfs
        dfs = {}
        for spec in specie:
            
            df_spec = df[df['specie'] == specie[spec]]   
            timestamps = df_spec['time'].unique()
            animals = df_spec['animal'].unique()   
            df_quant = None
            df_quant = pd.DataFrame(index = timestamps,columns = animals)
            
            for i in animals:
                temp_df = df_spec[df_spec['animal'] == i]
                df_quant[i].loc[temp_df['time']] = temp_df['midtime'].values
            
            dfs.update({spec:df_quant.fillna(0)})
        
    return dfs
 
    
def dataplot_mark_dopage(axe,date_range):
    """
    Shade the doping period and mark the doping moment thoroughly
    """
    #shade over doping period - item extracts value from pandas series
    axe.axvspan(date_range[0], date_range[1], alpha=0.7, color='orange')
    
    #plot vertical line at estimated moment of dopage
    #could add dopage as function parameter
    axe.axvline(date_range[1], color = 'red')
    
    #could add possibility to put xtick in location of doping?
    
def rolling_mean(df,timestep):
    
    #convert mins to number of 20 second intervals
    timestep = (timestep * 60)//20
    return df.rolling(timestep).mean().dropna()

def read_mapping(Tox,rootdate = 20220225,remapping = False):
    
    #configroot i.e. r'I:\Shared\Configs\Mappings\769'
    #rootdate of datafile not mapping i.e. 20210702 from I:\TXM762-PC\20210702-082335
    #rootdate unless specified assumed tobe recent
    
    mapping = {}
    configroot = r'I:\Shared\Configs\Mappings\{}'.format(Tox)
    if remapping: configroot = r'I:\Shared\Configs\ReMappings\{}'.format(Tox)
    mappings = os.listdir(configroot)
    configpath = None
    
    if len(mappings) == 1:
        configpath = r'{}\{}'.format(configroot,mappings[0])
                
    else:        
        #loop from most recent date
        for m in mappings[::-1][1:]:
            mapdate = int(m.split('.')[0])
            if int(rootdate) > mapdate:
                configpath = r'{}\{}'.format(configroot,m)
                break
            
    if not configpath: configpath = r'{}\{}'.format(configroot,mappings[0])
                
    #read three line csv file containing true mapping order
    with open(configpath, 'r') as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            mapping.update({row[0]:[int(x) for x in row[1:]]})
                
    return mapping
    
def check_mapping(dfs,mapping):
    
    #check all values
    if type(dfs) == dict:
        for s in dfs:
            if mapping[s] == [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]: 
                pass
            else:
                df = pd.DataFrame(index = dfs[s].index)
                for i,x in enumerate(mapping[s]):
                    df[x] = dfs[s][i+1]
                df = df[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]]
                dfs[s] = df
            
        return dfs
    
    #case of single df (mapping should be list)          
    else:
        if mapping == [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]: 
            df = dfs
            pass
        else:
            df = pd.DataFrame(index = dfs.index)
            for i,x in enumerate(mapping):
                df[x] = dfs[i+1]
            df = df[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]]
                
        return df
    
    
def remove_dead(df,species):
    
    """
    If there are very few entries, remove with different strategy
    """
    
    if df.shape[0] < 3000:
        remove = []
        for col in df.columns:
            #relies of having at least one value that recurs ! USE iloc instead
            ratio = df[col].value_counts().iloc[0]/df.shape[0]
            if ratio > 0.95 : remove.append(col)
            
        return remove
        
        
    else:
        #maybe this isnt even necessary
        threshold_dead = {'G':1000,'E':1000,'R':2000}
        
        #assume all organisms that die should be removed completely
        max_counts = []
        for col in df.columns:
            
            # find max zero counter excluding single peaks
            counter = 0
            max_count = 0
            for i in range(1,df.shape[0]):
                if df[col].iloc[i]:
                    if df[col].iloc[i-1]:
                        if counter > max_count: max_count = counter
                        counter = 0
                else:
                    counter += 1
                
            if counter > max_count: max_count = counter
            
            max_counts.append(max_count)
            
        print(max_counts)
        return [col+1 for col,val in enumerate(max_counts) if val > threshold_dead[species]]

def check_dead(df,species):
    threshold_dead = {'G':1000,'E':1000,'R':2000}
    max_counts = []
    for col in df.columns:
        
        # find max zero counter excluding single peaks
        counter = 0
        max_count = 0
        for i in range(1,df.shape[0]):
            if df[col].iloc[i]:
                if df[col].iloc[i-1]:
                    #add max count if it is greater than current
                    if counter > max_count: max_count = counter
                    counter = 0
            else:
                counter += 1
            
        if counter > max_count: max_count = counter
        
        max_counts.append(max_count)
   
    return [df.columns[i] for i,val in enumerate(max_counts) if val > threshold_dead[species]]
    
def write_csv(df,s,root):
    #write as zip file to math format in NAS
    return None
    
def remove_dead_known(dfs,morts):
    
    for s in dfs:
        dfs[s] = dfs[s].drop(columns = morts[s])
    return dfs
            
def gendirs(et_no):
    
    """
    Each week use this function to create relevant directories for results
    """
    
    et = study_no(et_no)
    
    root = os.getcwd()
    os.chdir(r'D:\VP\Viewpoint_data')
    
    for i in range(0,10):
        os.chdir('TxM76{}-PC'.format(i))
        os.mkdir('{}'.format(et))
        os.chdir('..')
        
    os.chdir(root)

def unzipfiles(et_no):
    
    et = study_no(et_no)
    root = os.getcwd()
    
    #loop through and unzip
    for i in range(0,10):
        os.chdir(r'D:\VP\Viewpoint_data\TxM76{}-PC\{}'.format(i,et))
        for file in [f for f in os.listdir() if '.zip' in f]:
            with zipfile.ZipFile(file) as item: item.extractall()
            os.remove(file)
            print('TxM76{} - Unzipped'.format(i))
        
    os.chdir(root)

def savefig(name, fig):
    os.chdir('Results')
    fig.savefig(name)
    os.chdir('..')

def dope_params(df, Tox, start_date, end_date):
    
    #do not assume that the date is on a friday
    dope_df = df[(df['End'] > start_date) & (df['End'] < end_date)]
    dope_df = dope_df[dope_df['TxM'] == Tox]
    
    if dope_df.shape[0] == 1:
        date_range = [
            dope_df['Start'].iloc[0],
            dope_df['End'].iloc[0]
            ]
        conc = dope_df['Concentration'].iloc[0]
        etude = df[(df['TxM'] == Tox) & (df['Start'] == date_range[0])].index[0]//5 + 1
        
    else:
        date_range = [
            dope_df['Start'],
            dope_df['End']
            ]
        conc = dope_df['Concentration']
        etude = df[(df['TxM'] == Tox) & (df['Start'] == date_range[0].iloc[0])].index[0]//5 + 1
        
    dopage = date_range[1]
    sub = dope_df['Substance'].iloc[0]
    molecule = dope_df['Molecule'].iloc[0]
    
    """
    etude is the number of the week of the experiment.
    Etude1 would be my first week of experiments
    """
    return dopage,date_range,conc,sub,molecule,etude

def plot_16(df,title  = '',mark = None):
    
    """
    Plot a 16 square subplots - assuming columns from 1-16
    """
    fig,axe = plt.subplots(4,4,sharex = True,sharey = True,figsize = (20,8))
    plt.suptitle(title)
    for i in range(1,17):
        if i in df.columns:
            axe[(i-1)//4,(i-1)%4].plot(df.index,df[i],color = colors[2])
            if mark:
                dataplot_mark_dopage(axe[(i-1)//4,(i-1)%4],mark)
        axe[(i-1)//4,(i-1)%4].tick_params(axis='x', rotation=90)
        
    return fig,axe

def single_plot(series,title = '',ticks = None):
    fig = plt.figure(figsize = (13,8))
    axe = fig.add_axes([0.1,0.1,0.8,0.8])
    axe.plot(series.index,series)
    axe.set_title(title)
    axe.tick_params(axis = 'x', rotation = 90)
    # if ticks:
    #     axe.set_xticks(ticks)
    #     axe.set_xticklabels([xticklabel_gen(i) for i in ticks])
    #     plt.xticks(rotation = 15)
        
    return fig,axe

def single_plot16(df,species,title = '',ticks = None):
    fig = plt.figure(figsize = (13,8))
    axe = fig.add_axes([0.1,0.1,0.8,0.8])
    for i in df.columns:
        axe.plot(df.index,df[i],label = '{}{}'.format(species,i),color = colors[i-1])
    axe.tick_params(axis = 'x', rotation = 90)
    
    # #xticks
    # if ticks:
    #     axe.set_xticks(ticks)
    #     axe.set_xticklabels([xticklabel_gen(i) for i in ticks])
    #     plt.xticks(rotation = 15)
        
    #title    
    axe.set_title(title)
    plt.legend()
    
    return fig,axe


# Covert date for corrupt VPCore2 files
def study_no(etude):
    if type(etude) == int:
        if etude <10:
            etude = 'Etude00{}'.format(etude)
        elif etude < 100:
            etude = 'Etude0{}'.format(etude)
        else:
            etude = 'Etude{}'.format(etude)
            
        return etude
    else:
        return etude

def distplot(df,species = 'G'):
    
    plt.figure()
    plt.xlim(0,1000)
    df[df != 0].stack().hist(bins = 500)
    
    thresh_map = {'E':190,'G':250,'R':80}
    upper_threshold = thresh_map[species]
    plt.axvline(x = upper_threshold, color = 'red')
    
    zeroed = int(100*((df == 0).sum().sum())/df.size)
    centred = int(100*len(df[(df != 0) & (df < upper_threshold)].stack())/df.size)
    upper = int(100*len(df[df > upper_threshold].stack())/df.size)
    
    plt.title('Valeurs nulles: {}%, Valeurs 1-200: {}%, Valeurs 200+: {}%'.format(zeroed,centred,upper)) 


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

def read_calibrationscale(Tox,start):
    """
    Read relevant scales
    
    Read calibrations after given date for input date (start)
    Return most recent after date, or last in list
    """
    calibrations = pd.read_csv(r'D:\VP\Viewpoint_data\REGS\calibrationscales.csv',header = None)
    calibrations[1] = pd.to_datetime(calibrations[1],format = '%Y%m%d')
    calibrations = calibrations[calibrations[0] == Tox]
    
    #dates are one out of shift, select most recent after date
    try:
        return np.array(calibrations[calibrations[1] > start].iloc[0][2:])
    except:
        return np.array(calibrations.iloc[-1][2:])
    
def calibrate(dfs,Tox,start):
    """ Use read calibration to correct dfs """
    calibration = read_calibrationscale(Tox, start)
    for i,s in enumerate(dfs):
        dfs[s] = dfs[s]/calibration[i]
    return dfs

def read_starttime(root):
    """ read datetime and convert to object from txt file """
    startfile = open(r'{}\start.txt'.format(root),"r")
    starttime = startfile.read()
    startfile.close()
    return datetime.strptime(starttime,'%d/%m/%Y %H:%M:%S')

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

def correct_index(df,start,correction = 0.997):
    """ Account for time warp error in video generation """
    ind = np.array((df.index - df.index[0]).total_seconds() * correction)
    ind = pd.to_datetime(ind*pd.Timedelta(1,unit = 's') + pd.to_datetime(start))
    df_ = df.copy()
    df_.index = ind
    return df_

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