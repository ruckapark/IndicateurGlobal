# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 15:11:53 2021

Selection of moving mean for IGT performance in Gammare

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

def dataplot_time():
        fig = plt.figure(figsize = (14,7), dpi = 100)
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
    df[df['dist']>0]['dist'].hist(bins = 1000)
    
    thresh_map = {'E':180,'G':200,'R':50}
    upper_threshold = thresh_map[species]
    plt.axvline(x = upper_threshold, color = 'red')
    
    zeroed = int(100*len(df[(df.dist == 0)])/len(df))
    centred = int(100*len(df[(df.dist > 0) & (df.dist < upper_threshold)])/len(df))
    upper = int(100*len(df[df.dist > upper_threshold])/len(df))
    
    plt.title('Valeurs nulles: {}%, Valeurs 1-200: {}%, Valeurs 200+: {}%'.format(zeroed,centred,upper))
    
    #add text for distribution in lar, sml and ina
    ina_100 = len(df[df['inadur']>0])/len(df) * 100
    sml_100 = len(df[df['smldur']>0])/len(df) * 100
    lar_100 = len(df[df['lardur']>0])/len(df) * 100
    empty_100 = len(df[df['emptydur']>0])/len(df) * 100
    
    #figure y coordinate for y text according to species - not going to add to 100...
    y_label_coor = {'E':4000,'G':4000,'R':4000}
    text = 'ina: {} \nsml: {} \nlar: {} \nempty {}'.format(ina_100,sml_100,lar_100,empty_100)
    plt.text(700,y_label_coor[species],text)    
    
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
   
def study_no(etude):
    if etude <10:
        etude = 'ETUDE_00{}'.format(etude)
    elif etude < 100:
        etude = 'ETUDE_0{}'.format(etude)
    else:
        etude = 'ETUDE_{}'.format(etude)
        
    return etude

#%% SETUP

sns.set_style('darkgrid', {"xtick.major.size": 8, "ytick.major.size": 8})
sns.set_context('paper')
colors = [
    '#42f5e0','#12aefc','#1612fc','#6a00a3',
    '#8ef743','#3c8f01','#0a4001','#fc03ca',
    '#d9d200','#d96c00','#942c00','#fc2803',
    '#e089b6','#a3a3a3','#7a7a7a','#303030'
    ]
       
Tox,species = 764,'G'
specie = {'E': 'Erpobdella','G':'Gammarus','R':'Radix'}

#os.chdir(r'D:\VP\Viewpoint_data\TxM{}-PC'.format(Tox))
os.chdir(r'D:\VP\Viewpoint_data\IGT_dev_gamm')
files = [file for file in os.listdir() if os.path.isfile(file)]

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

#%% plot
    
date_dopage = dfs[-1].iloc[0]['time'].strftime('%d/%m/%Y')
dope_df = dope_read()

# dopage is exact datetime
dopage,date_range,conc,sub,molecule,etude = dope_params(dope_df,Tox,date_dopage)

# add experiment columns in
df['dose'] = conc
df['etude'] = study_no(etude)
df['lot'] = sub
df['formule'] = molecule


# could also seperate out over time
distplot(df)
 
timestamps = df['time'].unique()
animals = list(range(1,17))   
df_dist = pd.DataFrame(index = timestamps)

for i in animals:
    temp_df = df[df['animal'] == i]
    df_dist[i] = temp_df['dist'].values
    

### test IGT mean selection value
moving_means = [2*i for i in range(90)]
resolution = []

df_morts = df_dist.rolling(400).std().dropna()
morts_test = list(df_morts.min()[df_morts.min() < 1].index)


# for some reason this ruins the IGT
for i in morts_test:
    # df_dist[i] = df_dist[i].where(df_morts[i] < 1)
    df_dist[i][df_dist.index > df_morts[df_morts[i] < 1].index[0]] = np.nan
    
    
# morts = [2,11]
# df_dist = df_dist.drop(morts_test, axis = 1)
    
"""
Find the best resolution for the moving means
"""


for i in moving_means:
    timestep = i
    mean_window = (timestep * 60)//20
    
    #get IGT for given sliding window
    if mean_window == 0:
        df_mean_dist = df_dist
    else:
        df_mean_dist = df_dist.rolling(mean_window).mean().dropna()
    quantile_dist = df_mean_dist.quantile(q = 0.05, axis = 1)**2
    
    #find pre dopage max value (minus 30 mins for margin of error)
    pre_dope_max = quantile_dist[quantile_dist.index < (dopage - pd.Timedelta(minutes = 30))].max()
    post_dope_max = quantile_dist[quantile_dist.index > (dopage - pd.Timedelta(minutes = 2))].max()
    
    
    resolution.append(100*(post_dope_max/pre_dope_max - 1))
    
    if i % 10 == 0:
        
        no_xticks = 10
        xticks = [df_mean_dist.index[i*len(df_mean_dist)//no_xticks] for i in range(no_xticks)]
        
        fig_IGT,axe_IGT = dataplot_time()
        axe_IGT.plot(quantile_dist.index,quantile_dist)
        dataplot_time_params(
            axe_IGT,
            '{} - IGT    {}   {}   {}      Moving mean:{}mins'.format(specie[species],sub,conc,etude,timestep),
            xticks,
            ylab = 'IGT')
        dataplot_mark_dopage(axe_IGT,date_range)
        
plt.figure()
plt.plot(moving_means,resolution)