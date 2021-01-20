# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 16:19:24 2020

Sortir graphiques didier

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


#%% FUNCTIONS


#%% SETUP

sns.set_style('darkgrid', {"xtick.major.size": 8, "ytick.major.size": 8})
sns.set_context('paper')
colors = [
    '#42f5e0','#12aefc','#1612fc','#6a00a3',
    '#8ef743','#3c8f01','#0a4001','#fc03ca',
    '#d9d200','#d96c00','#942c00','#fc2803',
    '#e089b6','#a3a3a3','#7a7a7a','#303030'
    ]
       
Tox,species = 764,'R'
specie = {'E': 'Erpobdella','G':'Gammarus','R':'Radix'}

#os.chdir(r'D:\VP\Viewpoint_data\TxM{}-PC'.format(Tox))
os.chdir(r'D:\VP\Viewpoint_data\temp')
files = [file for file in os.listdir() if os.path.isfile(file)]

#create a results folder if necessary


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

#create animal column from location E01 -> 1 E12 -> 12
df['animal'] = df['location'].str[1:].astype(int)

# doesn't appear necessary - what is it?
df = df.drop('entct',axis = 1)

#%% data manipulation
"""
Datetime is in nanoseconds, //10e9 to get seconds
Can zero this value
"""
#recreate abtime as seconds since first value - For some reason creates a huge diff day to day
df['abtime'] = df['time'].astype('int64')//1e9 #convert nano
df['abtime'] = df['abtime'] - df['abtime'][0]

#total distance inadist is only zeros?
df['dist'] = df['inadist'] + df['smldist'] + df['lardist']

#species is input
df = df[df['specie'] == specie[species]]

#%% FUNCTIONS

def dataplot_time():
        fig = plt.figure(figsize = (8,5), dpi = 100)
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
        axe.axvspan(date_range[0]-pd.Timedelta(minutes = 10), date_range[1]-pd.Timedelta(minutes = 10), alpha=0.7, color='orange')
        
        #plot vertical line at estimated moment of dopage
        #could add dopage as function parameter
        axe.axvline(date_range[0]-pd.Timedelta(minutes = 10) + pd.Timedelta(seconds = 90), color = 'red', linestyle = '--')
    
    #could add possibility to put xtick in location of doping?

def distplot(df):
    plt.figure()
    plt.xlim(0,1000)
    df[df['dist']>0]['dist'].hist(bins = 1000)
    plt.axvline(x = 200, color = 'red')
    plt.title('Valeurs nulles: {}%, Valeurs 1-200: {}%, Valeurs 200+: {}%'.format(int(100*len(df[(df.dist == 0)])/len(df)),int(100*len(df[(df.dist > 0) & (df.dist < 200)])/len(df)),int(100*len(df[df.dist > 200])/len(df))))
    
def savefig(name, fig):
    os.chdir('Results')
    fig.savefig(name)
    os.chdir('..')

def dope_params(df, Tox, date):
    
    date_range = [
        df[(df['TxM'] == Tox) & (df['Start'].dt.strftime('%d/%m/%Y') == date)]['Start'].values[0],
        df[(df['TxM'] == Tox) & (df['End'].dt.strftime('%d/%m/%Y') == date)]['End'].values[0]
        ]
    dopage = date_range[0] + pd.Timedelta(seconds = 90)
    
    # row value of experiment in dope reg
    conc = df[(dope_df['TxM'] == Tox) & (dope_df['Start'].dt.strftime('%d/%m/%Y') == date)]['Concentration'].values[0]
    sub = df[(dope_df['TxM'] == Tox) & (dope_df['Start'].dt.strftime('%d/%m/%Y') == date)]['Substance'].values[0]
    molecule = df[(dope_df['TxM'] == Tox) & (dope_df['Start'].dt.strftime('%d/%m/%Y') == date)]['Molecule'].values[0]
    
    """
    etude is the number of the week of the experiment.
    Etude1 would be my first week of experiments
    """
    etude = df[(dope_df['TxM'] == Tox) & (dope_df['Start'].dt.strftime('%d/%m/%Y') == date)].index[0]//5 + 1
    return dopage,date_range,conc,sub,molecule,etude
    
#%% plot
    
date_dopage = dfs[-1].iloc[0]['time'].strftime('%d/%m/%Y')
dope_df = dope_read()

# dopage is exact date of dopage precise to the minute
dopage,date_range,conc,sub,molecule,etude = dope_params(dope_df,Tox,date_dopage)

# add experiment columns in
df['dose'] = conc
df['etude'] = 'ETUDE{}'.format(etude)
df['lot'] = sub
df['formule'] = molecule


# could also seperate out over time
distplot(df)
    
"""
all time data
"""

fig,axe = dataplot_time()
for i in range(1,17):
    axe.plot(df[df['animal']==i]['time'], df[df['animal']==i]['dist'], label = i, color = colors[i-1])

no_xticks = 10
xticks = [df.iloc[i*len(df)//no_xticks]['time'] for i in range(no_xticks)]

dataplot_time_params(
    axe,
    '20s distance - {}'.format(specie[species]),
    xticks,
    ylab = 'Distance',
    ylim = [0,df['dist'].quantile(0.9999)],
    text = '{} \n{} \n{}'.format(sub,conc,molecule)
    )

dataplot_mark_dopage(axe,date_range)


#%% generate
"""
df with distances, each individual in column
"""

# this does not work
timestep_min = 10
timestep = timestep_min*60 # seconds
df['timestep'] = df['abtime']//timestep * timestep


timesteps = df['timestep'].unique().astype(int)
timestamps = [df[df['timestep'] == timestep]['time'].iloc[0] for timestep in timesteps]

animals = list(range(1,17))

#append mean distance 1 by 1
df_mean_dist = pd.DataFrame(index = timestamps)

#groupby animal method? - create df with column as cell and mean distances
#by making this from the means, the cadre mean and IGT is calculated from a mean ...
for i in animals:
    temp_df = df[df['animal'] == i]
    mean_distance = temp_df.groupby(['timestep']).mean()['dist']
    #use .values to avoid index errors and disassociate
    df_mean_dist[i] = mean_distance.values
    
#%% plot
"""
mean individual distances
"""

fig_means,axe_means = dataplot_time()
for i in range(1,17):
    axe_means.plot(df_mean_dist.index,df_mean_dist[i], label = i, color = colors[i-1])

no_xticks = 10
xticks = [df.iloc[i*len(df)//no_xticks]['time'] for i in range(no_xticks)]

dataplot_time_params(
    axe_means,
    '{}s mean distance - {}'.format(timestep,specie[species]),
    xticks,
    ylab = 'Distance'
    )

dataplot_mark_dopage(axe_means,date_range)

#%% calculate
"""
population mean
IGT
"""

#plot all the means across cells
mean_dist = df_mean_dist.mean(axis = 1)
quantile_dist = df_mean_dist.quantile(q = 0.05, axis = 1)**2

#%%
"""
population mean plot
IGT plot
"""

fig_mean_all,axe_mean_all = dataplot_time()
axe_mean_all.plot(mean_dist.index,mean_dist)

dataplot_time_params(
    axe_mean_all,
    '{} - Mean cadre'.format(specie[species]),
    xticks,
    ylab = 'Distance')
dataplot_mark_dopage(axe_mean_all,date_range)

#############

fig_IGT,axe_IGT = dataplot_time()
axe_IGT.plot(quantile_dist.index,quantile_dist)
dataplot_time_params(
    axe_IGT,
    '{} - IGT    {}   {}   {}'.format(specie[species],sub,conc,etude),
    xticks,
    ylab = 'IGT')
dataplot_mark_dopage(axe_IGT,date_range)

############

"""
plot gammares 1 by 1 on a 16 by 16 subplot - how to use sharey ?
"""
fig_indi, axs_indi = plt.subplots(4,4,sharex = True, sharey = True, figsize = (18,14))
fig_indi.suptitle('Distances individuelles')
means = []
for i in animals:
    ax_r,ax_c = (i-1)//4, (i-1)%4
    axs_indi[ax_r,ax_c].plot(df[df['animal']==i]['time'], df[df['animal']==i]['dist'], color = colors[i-1])
    axs_indi[ax_r,ax_c].set_title(i)
    means.append(df[df['animal']==i]['dist'].mean())
    
means = pd.Series(means)    

#%% calculate
"""
redo taking out dead gammares
"""
#only take upto 1 hour after the dopage?
df_morts = df_mean_dist[df_mean_dist.index < (dopage + pd.Timedelta(hours = 1))].rolling(100).mean().dropna()
morts = list(df_morts.min()[df_morts.min() < 1].index)

# plot all the means across cells
# morts = [2,11]
df_mean_dist = df_mean_dist.drop([m for m in morts], axis = 1)
for m in morts[::-1]:
    animals.pop(m-1)

mean_dist = df_mean_dist.mean(axis = 1)
quantile_dist = df_mean_dist.quantile(q = 0.05, axis = 1)**2
quantile_dist[quantile_dist.index < dopage - pd.Timedelta(minutes = 15)] = 0

fig_mean_all,axe_mean_all = dataplot_time()
axe_mean_all.plot(mean_dist.index,mean_dist)

dataplot_time_params(
    axe_mean_all,
    '{} - Mean cadre'.format(specie[species]),
    xticks,
    ylab = 'Distance')
dataplot_mark_dopage(axe_mean_all,date_range)

#############

fig_IGT,axe_IGT = dataplot_time()
axe_IGT.plot(quantile_dist.index,quantile_dist.rolling(8).mean().fillna(0))
# dataplot_time_params(
#     axe_IGT,
#     '{} - IGT    {}   {}   {}'.format(specie[species],sub,conc,etude),
#     xticks,
#     ylab = 'IGT')
dataplot_mark_dopage(axe_IGT,date_range)

############