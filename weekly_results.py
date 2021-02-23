# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 10:10:12 2021

File created for the weekly run of visualising data

@author: Admin
"""

#%% IMPORTS

import os
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from datetime import timedelta
from data_merge import merge_dfs
from dope_reg import dope_read


#%% FUNCTIONS - plotting functions could also be put into another module

def plot_16(df):
    
    """
    Plot a 16 square subplots
    """
    fig,axe = plt.subplots(4,4,sharex = True, figsize = (20,12))
    for i in df.columns:
        axe[(i-1)//4,(i-1)%4].plot(df.index,df[i],color = colors[i-1])
        axe[(i-1)//4,(i-1)%4].tick_params(axis='x', rotation=90)
        
    return fig,axe

def single_plot(series,title = '',ticks = None):
    fig = plt.figure(figsize = (13,8))
    axe = fig.add_axes([0.1,0.1,0.8,0.8])
    axe.plot(series.index,series)
    axe.set_title(title)
    if ticks:
        axe.set_xticks(ticks)
        axe.set_xticklabels([xticklabel_gen(i) for i in ticks])
        plt.xticks(rotation = 15)
        
    return fig,axe

def single_plot16(df,species,title = '',ticks = None):
    fig = plt.figure(figsize = (13,8))
    axe = fig.add_axes([0.1,0.1,0.8,0.8])
    for i in df.columns:
        axe.plot(df.index,df[i],label = '{}{}'.format(species,i),color = colors[i-1])
    
    #xticks
    if ticks:
        axe.set_xticks(ticks)
        axe.set_xticklabels([xticklabel_gen(i) for i in ticks])
        plt.xticks(rotation = 15)
        
    #title    
    axe.set_title(title)
    plt.legend()
    
    return fig,axe

def dataplot_time():
    fig = plt.figure(figsize = (20,10), dpi = 100)
    axe = fig.add_axes([0.1,0.1,0.8,0.8])
    return fig,axe

def xticklabel_gen(timestamp,lang = 'fr'):
    """
    return from timestamp weekday and time HH:MM xtick_label
    
    change lang if we want dates in english
    """
    if lang == 'fr':
        days = dict(zip([0,1,2,3,4,5,6],'lu ma me je ve sa di'.split()))
    else:
        days = dict(zip([0,1,2,3,4,5,6],'mo tu we th fr sa su'.split()))
    
    wday = days[timestamp.weekday()]
    hour = timestamp.hour
    minute = timestamp.minute
    if hour < 10:
        hour = '0'+ str(hour)
    else:
        hour = str(hour)
        
    if minute < 10:
        minute = '0'+str(minute)
    else:
        minute = str(minute)
        
    return wday + '-' + hour + ':' + minute

def dataplot_time_params(axe,title,xticks,ylab = None,xlab = None,
                             ylim = None,xlim = None, rot = 10, text = None
                             ):
    axe.set_title(title)
    axe.legend()            #if no label pas grave
    axe.set_ylabel(ylab)
    axe.set_xlabel(xlab)
    if ylim: axe.set_ylim(ylim[0],ylim[1])
    if xlim: axe.set_xlim(xlim[0],xlim[1])    
    axe.set_xticks(xticks)
    axe.set_xticklabels([xticklabel_gen(i) for i in xticks])
    plt.xticks(rotation = rot)
    # doesnt work
    if text: axe.text(0.75,0.75,text,fontsize = 9, bbox = dict(boxstyle = 'round', alpha = 0.6))
    
def dataplot_mark_dopage(axe,date_range):
    """
    Shade the doping period and mark the doping moment thoroughly
    """
    #shade over doping period - item extracts value from pandas series
    axe.axvspan(date_range[0], date_range[1], alpha=0.7, color='orange')
    
    #plot vertical line at estimated moment of dopage
    #could add dopage as function parameter
    axe.axvline(date_range[0] + pd.Timedelta(seconds = 90), color = 'red')
    
    #could add possibility to put xtick in location of doping?

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
    
def savefig(name, fig):
    os.chdir('Results')
    fig.savefig(name)
    os.chdir('..')

def dope_params(df, Tox, start_date, end_date):
    
    #do not assume that the date is on a friday
    dope_df = df[(df['Start'] > start_date) & (df['Start'] < end_date)]
    dope_df = dope_df[dope_df['TxM'] == Tox]
    
    date_range = [
        dope_df['Start'].iloc[0],
        dope_df['End'].iloc[0]
        ]
    dopage = date_range[0] + pd.Timedelta(seconds = 90)
    
    # row value of experiment in dope reg
    conc = dope_df['Concentration'].iloc[0]
    sub = dope_df['Substance'].iloc[0]
    molecule = dope_df['Molecule'].iloc[0]
    
    """
    etude is the number of the week of the experiment.
    Etude1 would be my first week of experiments
    """
    
    #########################
    
    etude = df[(df['TxM'] == Tox) & (df['Start'] == date_range[0])].index[0]//5 + 1
    return dopage,date_range,conc,sub,molecule,etude
   
def study_no(etude):
    if etude <10:
        etude = 'ETUDE_00{}'.format(etude)
    elif etude < 100:
        etude = 'ETUDE_0{}'.format(etude)
    else:
        etude = 'ETUDE_{}'.format(etude)
        
    return etude

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
    
    
def rolling_mean(df,timestep):
    
    #convert mins to number of 20 second intervals
    timestep = (timestep * 60)//20
    return df.rolling(timestep).mean().dropna()
    

#%% SETUP

sns.set_style('darkgrid', {"xtick.major.size": 8, "ytick.major.size": 8})
sns.set_context('paper')
colors = [
    '#42f5e0','#12aefc','#1612fc','#6a00a3',
    '#8ef743','#3c8f01','#0a4001','#fc03ca',
    '#d9d200','#d96c00','#942c00','#fc2803',
    '#e089b6','#a3a3a3','#7a7a7a','#303030'
    ]

   
Tox,species,etude_ = 763,'G','Etude009'
specie = {'E': 'Erpobdella','G':'Gammarus','R':'Radix'}

os.chdir(r'D:\VP\Viewpoint_data\TxM{}-PC\{}'.format(Tox,etude_))
files = [file for file in os.listdir() if os.path.isfile(file)]
 
df = read_merge(files)
dfs = preproc(df)

# could also seperate out over time
df = dfs[species]
distplot(df)

#%% plot

dope_df = dope_read()

# dopage is exact datetime
dopage,date_range,conc,sub,molecule,etude = dope_params(dope_df,Tox,df.index[0],df.index[-1])

no_xticks = 10
xticks = [df.index[i*len(df)//no_xticks] for i in range(no_xticks)]

fig,axe = single_plot16(df, species, title = '20s distance - {}'.format(specie[species]),ticks = xticks)
dataplot_mark_dopage(axe,date_range)


# moving means

t_mins = 5
df_mean = rolling_mean(df,t_mins)
fig,axe = single_plot16(df_mean, species, title = '{} Moving mean - {}'.format(t_mins,specie[species]), ticks = xticks[1:])
dataplot_mark_dopage(axe,date_range)
fig,axe = plot_16(df_mean)

#%% calculate
"""
population mean
IGT
"""

#plot all the means across cells
mean_dist = df_mean.mean(axis = 1)
quantile_dist = df_mean.quantile(q = 0.05, axis = 1)**2

fig,axe = single_plot(mean_dist,title = 'Mean distance',ticks = xticks[1:])
dataplot_mark_dopage(axe,date_range)

title_ = 'ToxIndex: {}({}), {}   {}-{}'.format(sub,molecule,conc,date_range[0].strftime('%d/%m'),date_range[1].strftime('%d/%m'))

fig,axe = single_plot(quantile_dist, title = title_, ticks = xticks[1:])
dataplot_mark_dopage(axe,date_range)

#2 hour
IGT = quantile_dist.loc[date_range[0] - pd.Timedelta(minutes = 20) : date_range[0] + pd.Timedelta(minutes = 90)]
xticks = [IGT.index[i*len(IGT)//no_xticks] for i in range(no_xticks)]
fig,axe = single_plot(IGT, title = title_, ticks = xticks)
axe.set_xlabel('Tox Ind')
dataplot_mark_dopage(axe,date_range)


"""
Nitrate - 765 13
Chlorure - 764 13
Alu - 763 9
Mercury - 763 11 -- NA
Zinc - 767 5
Manganese - 765 6
Cuivre - NA
"""