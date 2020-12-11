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
import plotly.express as px
import seaborn as sns
from datetime import timedelta

#from perftimer import Timer
#t = Timer()


os.chdir(r'D:\VP\Viewpoint_data\Exp_semaine_10_11\764\av_dope')
files = os.listdir()

#start with only df group only use 2500 lines
df = pd.read_csv(files[0],sep = '\t',encoding = 'utf-16')
df = df[df['datatype'] == 'Locomotion']

#sort values sn = , pn = ,location = E01-16 etcc., aname = A01-04,B01-04 etc.
df = df.sort_values(by = ['sn','pn','location','aname'])
df = df.reset_index(drop = True)

#treat time variable
df['time'] = pd.to_datetime(df['stdate'] + " " + df['sttime'])

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

isnull = df.isnull().sum()
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
#recreate abtime as seconds since first value
df['abtime'] = df['time'].astype('int64')//1e9 #convert nano
df['abtime'] = df['abtime'] - df['abtime'][0]

#create threshold columns
df['threshold'] = 10
df['thresholdHigh'] = 20
df['thresholdLow'] = 5
df['protocole'] = 1

#%% IGT

start = pd.Timestamp('2020-10-11 15:17:53')
# stop = timedelta or whatever

#only go to value divisible by 48
maxrows = len(df)//48
if len(df) != maxrows*48:
    df = df.iloc[:maxrows*48]
    
#gammarus - make function where we input the specie
df = df[df['specie']=='Gammarus']

#total distance
df['dist'] = df['inadist'] + df['smldist'] + df['lardist']

#plot all animals
fig = px.line(df, x = 'time', y = 'dist', color = 'animal')
fig.show()

"""
#timestep in minutes for plotting interval (sometimes they only take 20 minutes)
timestep = (1/3) * 60 #mins converted to secs
timesteps = [i for i in range(df[df'time' == start]['abtime'],df[df'time' == stop]['abtime'],timestep)]

#extract only values in timesteps (missing information... - why not moyenne glissante)
df = df[df['abtime'].isin(timesteps)]
"""

#take the mean upto the first timestep for that cell
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
fig = px.line(df_mean_dist, x = df_mean_dist.index,y = df_mean_dist.columns)
fig.show()

#plot all the means across cells
mean_dist = df_mean_dist.mean(axis = 1)
fig_mean = px.line(mean_dist, x = mean_dist.index, y = mean_dist)
fig_mean.show()

#plot quantile 0.05 across cells
quantile_dist = df_mean_dist.quantile(q = 0.05, axis = 1)**2
fig_quant = px.line(mean_dist, x = mean_dist.index, y = mean_dist)
fig_quant.show()

# plot both in seperate figures


#%%
"""
fig = px.line(df, x = 'time', y = 'dist', color = 'animal')
fig.show()
"""
## pre data treatment
#1. split into locomotion and quantization
# print('Groupby method:')
# t.start()
# df_loc,df_quant = [g for _, g in df.groupby('datatype')]
# t.stop()

#should check performance over the full dataset - the other method is more readable
"""
print('Explicit method:')
t.start()
df_loc = df[df['datatype']=='Locomotion']
df_quant = df[df['datatype']=='Quantization']
t.stop()
"""

#reorder? - all values present but not all in the correct order

#3. split into animal type (asign 0 - radix, 1 - gammares,2 - sangsues)


#4.

#%%
os.chdir('..')