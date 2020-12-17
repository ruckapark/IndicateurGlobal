# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 13:56:07 2020

Steps:
    - increase figure size
    - rotate xtick labels
    - decrease font size
    - format - day and time eg. l,ma,me,j,v,s,d 09:13
    - add labels for multigammare plot
    - add y axe label
    - add x axe label
    - it seems at this stage plotly express is not useful.
    - set y limits to include 95% of the distributions values

@author: Admin
"""

#%% IMPORTS

import os
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
from data_merge import merge_dfs
from dope_reg import dope_read


#%% FUNCTIONS


#%% SETUP

sns.set_style('darkgrid', {"xtick.major.size": 8, "ytick.major.size": 8})
sns.set_context('paper')

Tox,species = 767,'G'
    
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

#%%

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
                         ylim = None,xlim = None, rot = 10
                         ):
    axe.set_title(title)
    axe.legend()
    axe.set_ylabel(ylab)
    axe.set_xlabel(xlab)
    if ylim: axe.set_ylim(ylim[0],ylim[1])
    if xlim: axe.set_xlim(xlim[0],xlim[1])    
    axe.set_xticks(xticks)
    axe.set_xticklabels([xticklabel_gen(i) for i in xticks])
    plt.xticks(rotation = rot)
    
def dataplot_mark_dopage(axe,dope_df,date,Tox):
    """
    Shade the doping period and mark the doping moment thoroughly
    """
    date_range = [
    dope_df[(dope_df['TxM'] == Tox) & (dope_df['Start'].dt.strftime('%d/%m/%Y') == date_dopage)]['Start'],
    dope_df[(dope_df['TxM'] == Tox) & (dope_df['End'].dt.strftime('%d/%m/%Y') == date_dopage)]['End']
    ]
    
    #shade over doping period - item extracts value from pandas series
    axe.axvspan(date_range[0].item(), date_range[1].item(), alpha=0.7, color='orange')
    
    #plot vertical line at estimated moment of dopage
    axe.axvline(date_range[0].item() + pd.Timedelta(minutes = 1), color = 'red')
    
    #could add possibility to put xtick in location of doping?

    

fig,axe = dataplot_time()
for i in range(1,17):
    axe.plot(df[df['animal']==i]['time'], df[df['animal']==i]['dist'], label = i)

no_xticks = 10
xticks = [df.iloc[i*len(df)//no_xticks]['time'] for i in range(no_xticks)]

dataplot_time_params(
    axe,
    '20s distance - {}'.format(specie[species]),
    xticks,
    ylab = 'Distance',
    ylim = [0,df['dist'].quantile(0.9999)]
    )

dope_df = dope_read()
date_dopage = dfs[-1].iloc[0]['time'].strftime('%d/%m/%Y')
dataplot_mark_dopage(axe,dope_df,date_dopage,Tox)


#plot all animals
fig = plt.figure(figsize = (20,10), dpi = 100)
axe = fig.add_axes([0.1,0.1,0.8,0.8])
for i in range(1,17):
    axe.plot(df[df['animal']==i]['time'], df[df['animal']==i]['dist'], label = i)
axe.set_title('Mean 20s distance: {}'.format(specie[species]))
axe.legend()
axe.set_ylabel('Distance')
axe.set_xlabel('Date (continuous adjustment)')
axe.set_ylim(0,df['dist'].quantile(0.9999))

# set xticklabels with certain number of xticks for days of the week
no_xticks = 10
xticks = [df.iloc[i*len(df)//no_xticks]['time'] for i in range(no_xticks)]
axe.set_xticks(xticks)
axe.set_xticklabels([xticklabel_gen(i) for i in xticks])
plt.xticks(rotation = 10)

#shade in doping in orange with a red line and the 40-60 seconds after interval
dope_df = dope_read()
date_dopage = dfs[-1].iloc[0]['time'].strftime('%d/%m/%Y')

#dataplot_mark_dopage(dope_df,datedopage)

##  DEBUG - occuring because 04/12/2020 -> 00:00:00. Therefore it happens again.
date_range = [
    dope_df[(dope_df['TxM'] == Tox) & (dope_df['Start'].dt.strftime('%d/%m/%Y') == date_dopage)]['Start'],
    dope_df[(dope_df['TxM'] == Tox) & (dope_df['End'].dt.strftime('%d/%m/%Y') == date_dopage)]['End']
    ]

#shade over doping period - item extracts value from pandas series
axe.axvspan(date_range[0].item(), date_range[1].item(), alpha=0.7, color='orange')

#plot vertical line at estimated moment of dopage
axe.axvline(date_range[0].item() + pd.Timedelta(minutes = 1), color = 'red')

#%%

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