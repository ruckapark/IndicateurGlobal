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

#from perftimer import Timer
#t = Timer()

os.chdir(r'D:\VP\Viewpoint_data\TxM767-PC')
files = os.listdir()

#start with only df group only use 2500 lines
df = pd.read_csv(files[1],sep = '\t',encoding = 'utf-16')
df = df[df['datatype'] == 'Locomotion']

#sort values sn = , pn = ,location = E01-16 etcc., aname = A01-04,B01-04 etc.
df = df.sort_values(by = ['sn','pn','location','aname'])
df = df.reset_index(drop = True)

#treat time variable - this gets the days and months the wrong way round
df['time'] = pd.to_datetime(df['stdate'] + " " + df['sttime'], format = '%d/%m/%Y %H:%M:%S')

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
#recreate abtime as seconds since first value - For some reason creates a huge diff day to day
df['abtime'] = df['time'].astype('int64')//1e9 #convert nano
df['abtime'] = df['abtime'] - df['abtime'][0]

#create threshold columns
df['threshold'] = 10
df['thresholdHigh'] = 20
df['thresholdLow'] = 5
df['protocole'] = 1

#%% IGT

# start = pd.Timestamp('2020-10-11 15:17:53')
# stop = timedelta or whatever

#only go to value divisible by 48
maxrows = len(df)//48
print('Before adjustment: total rows{}'.format(len(df)))
df = df.iloc[:maxrows*48]
print('After adjustment: total rows{}'.format(len(df)))

df = df[df['specie']=='Gammarus']

#total distance inadist is only zeros?
df['dist'] = df['inadist'] + df['smldist'] + df['lardist']

#plot all animals
fig = plt.figure()
axe = fig.add_axes([0.1,0.1,0.8,0.8])
for i in range(16):
    axe.plot(df[df['animal']==(i+1)]['time'],df[df['animal']==(i+1)]['dist'],label = 'gamm{}'.format(i+1))
axe.set_title('Mean values for Gammare')

"""
#timestep in minutes for plotting interval (sometimes they only take 20 minutes)
timestep = (1/3) * 60 #mins converted to secs
timesteps = [i for i in range(df[df'time' == start]['abtime'],df[df'time' == stop]['abtime'],timestep)]

#extract only values in timesteps (missing information... - why not moyenne glissante)
df = df[df['abtime'].isin(timesteps)]
"""

#take the mean upto the first timestep for that cell

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