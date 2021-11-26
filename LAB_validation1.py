# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 13:31:50 2021

All dataframes from the Validation are from the exact same recording.

@author: Admin
"""

#%% IMPORTS

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#%% FUNCTIONS - plotting functions could also be put into another module

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

def dataplot_time_params(axe,title,xticks = None,ylab = None,xlab = None,
                             ylim = None,xlim = None, rot = 10, text = None
                             ):
        axe.set_title(title)
        axe.legend()            #if no label pas grave
        axe.set_ylabel(ylab)
        axe.set_xlabel(xlab)
        if ylim: axe.set_ylim(ylim[0],ylim[1])
        if xlim: axe.set_xlim(xlim[0],xlim[1])    
        if xticks: 
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
    df[df['dist']>0]['dist'].hist(bins = 10)
    
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
os.chdir(r'D:\VP\Viewpoint_data\Validation')
files = [file for file in os.listdir() if (os.path.isfile(file) and ('.xls' in file))]

print('The following files will be analysed in this order:')
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
    
    timestamps = df['time'].unique()
    animals = list(range(1,17))   
    df_dist = pd.DataFrame(index = timestamps)

    for i in animals:
        df_dist[i] = df[df['animal'] == i]['dist'].values
    
    dfs.append(df_dist)

for i in animals:
    
    fig,axe = dataplot_time()
    
    for x in range(len(dfs)):
        df = dfs[x]
        axe.plot(df.index,df[i], label = files[x],color = colors[2*x])

    plt.legend()
    
    
fig, axe = plt.subplots(4,4,sharex = True, figsize = (20,12))
for i in animals:
    
    for x in range(len(dfs)):
        df = dfs[x]
        axe[(i-1)//4,(i-1)%4].plot(df.index,df[i], label = files[x],color = colors[2*x])

    plt.legend()
    fig.suptitle(specie[species])