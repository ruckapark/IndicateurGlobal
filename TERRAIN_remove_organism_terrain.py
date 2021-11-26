# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 15:28:37 2021

Find limits for the dopage terrain

@author: Admin
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta

#%% IMPORT personal mods
os.chdir('MODS')
from data_merge import merge_dfs
os.chdir('..')

root = r'D:\VP\Viewpoint_data\DATA_terrain'
os.chdir(root)
files = [f for f in os.listdir() if '.csv' in f]


colors = [
    '#42f5e0','#12aefc','#1612fc','#6a00a3',
    '#8ef743','#3c8f01','#0a4001','#fc03ca',
    '#d9d200','#d96c00','#942c00','#fc2803',
    '#e089b6','#a3a3a3','#7a7a7a','#303030'
    ]


specie = {'E': 'Erpobdella','G':'Gammarus','R':'Radix'}
species = 'G'

print('The following files will NOT be merged:')
print(files)

dfs = []

for file in files:
    df = pd.read_csv(file,sep = '\t')
    
    print('Erpo: {}'.format(len(df[df['specie'] == 'Erpobdella'])))
    print('Radix: {}'.format(len(df[df['specie'] == 'Radix'])))
    print('Gammare: {}'.format(len(df[df['specie'] == 'Gammarus'])))
    
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values(by = ['time','replica'])
    
    df = df.reset_index(drop = True)
    
    print('Before adjustment: total rows{}'.format(len(df)))
    
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
  
threshold = {'G':190,'R':50,'E':170}    

df = dfs[0]

df = df[df['specie'] == specie[species]]
    
df['animal'] = df['replica']

df['dist'] = df['inadist'] + df['smldist'] + df['lardist']
#df['dist'] = np.where((df['dist'] > threshold[species]),0,df['dist'])

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


# this does not work
timestep = 10*60    # 10 minute
start_time = df_dist.index[0]
df_dist['timestep'] = ((df_dist.index - start_time).total_seconds() // timestep).astype(int)

df_dist_mean = df_dist.groupby(['timestep']).mean()
index_map = dict(zip(df_dist['timestep'].unique(),[df_dist[df_dist['timestep'] == i].index[-1] for i in df_dist['timestep'].unique()]))
df_dist_mean = df_dist_mean.set_index(df_dist_mean.index.map(index_map))

fig_mean = plt.figure()
axe_mean = fig_mean.add_axes([0.1,0.1,0.8,0.8])
for i in animals:
    axe_mean.plot(df_dist_mean.index,df_dist_mean[i],color = colors[i-1],label = '{}{}'.format(species,i))
plt.legend()


fig,axe = plt.subplots(4,4,sharex = True, figsize = (20,12))
for i in animals:
    axe[(i-1)//4,(i-1)%4].plot(df_dist_mean.index,df_dist_mean[i],color = colors[i-1])
    
###################################    
#### Functions
    
def sub_df(df,day,cols):
    
    """
    return a day of dataframe as new df, with certain columns
    """
    return df[df.index.day == day][cols]    

def moving_zero_count(df,window_mins):
    
    """
    return number of zeros in a moving block - on axis 0 of dataframe
    """
    timesteps = window_mins*3
    df_zero_percentage = pd.DataFrame()
    for i in df.columns:
        zero_percentage = []
        for x in range(len(df)//(timesteps)):
            temp = df[i].iloc[(x*timesteps):((x+1)*timesteps)].values
            zero_percentage.append(len(temp[temp < 1])/timesteps)
        df_zero_percentage[i] = zero_percentage
        
    return df_zero_percentage

def plot_16(df,colors):
    
    """
    Plot a 16 square subplots
    """
    fig,axe = plt.subplots(4,4,sharex = True, figsize = (20,12))
    for i in df.columns:
        axe[(i-1)//4,(i-1)%4].plot(df.index,df[i],color = colors[i-1])
        axe[(i-1)//4,(i-1)%4].tick_params(axis='x', rotation=90)
        
    return fig,axe


def count_nonzero(df,xrange):
    
    """
    Return counts
    A count for a given x is the percentage of nonzero values above the x threshold
    """
    counts = []
    non_zero = (df > 0).values.sum()
    
    for x in xrange:
        value = (df > x).values.sum() / non_zero
        counts.append(value)
        
    return counts


def max_zero(df):
    
    """
    Max zeros that in a df
    """
    max_zer = []
    for i in df.columns:
        max_zer.append((df[i] != df[i].shift()).cumsum().value_counts().max())
    return max_zer


def delete_counter(vals, window = 1):
    
    """
    This function should interrupt and return dead if a certain threshold is reached
    
    - it doesn't quite work: 2,2,2,0 - starts counting at 2,2,0. Not at 0... Over a long time series this is unimportant
    """
    
    counter_final = 0
    counter = 0
    
    for i in range(len(vals) -window):
        if i < len(vals) - window - 1:
            if 0 not in vals[i:i+ window + 1]:
                if counter > counter_final: counter_final = counter
                counter = 0
            else:
                counter += 1
        else:
            if counter > counter_final: counter_final = counter
            
            
    return counter_final

def slide_max_zero(df):
    
    """
    list of slide max zeros
    """
    max_zero = []
    for i in df.columns:
        max_zero.append(delete_counter(list(df[i])))
        
    return max_zero
    
###################################
    
# separe df, vivant et mort
df_train_vivant = sub_df(df_dist, 13, animals)
df_train_mort = sub_df(df_dist, 15, animals)

# count zeros every (timestemp_mins) minutes
timestep_mins = 5
df_zero_percentage = moving_zero_count(df_train_vivant,timestep_mins)
df_zero_percentage2 = moving_zero_count(df_train_mort,timestep_mins)
    
# plot the data for the three curves
fig_vivant,axe_vivant = plot_16(df_train_vivant,colors)
fig_mort,axe_mort = plot_16(df_train_mort,colors)

df_tr_vi_nonactif = df_train_vivant[[12,13,15]]
df_tr_mo_actif = df_train_mort[[1,4,5]]

xrange = np.linspace(0,100,401)

count_vi = count_nonzero(df_train_vivant,xrange)
count_mo = count_nonzero(df_train_mort,xrange)
count_vi_nonactif = count_nonzero(df_tr_vi_nonactif,xrange)
count_mo_actif = count_nonzero(df_tr_mo_actif,xrange)

plt.figure()
plt.plot(xrange,count_vi,color = 'blue')
plt.plot(xrange,count_mo,color = 'red')
plt.plot(xrange,count_vi_nonactif, color = 'green')
plt.plot(xrange,count_mo_actif, color = 'pink')

#maximum zero dataframes
max_zero_vi = max_zero(df_train_vivant)
max_zero_mo = max_zero(df_train_mort)

max_zero3 = []
max_zero4 = []
for i in animals:
    max_zero3.append(delete_counter(list(df_train_vivant[i])))
    max_zero4.append(delete_counter(list(df_train_mort[i])))
    
    
# remove values below a threshold
df_threshold = pd.DataFrame(index = xrange, columns = ['Max vivant',
                                                       'Min mort',
                                                       'mean vivant',
                                                       'mean mort',
                                                       'q = 0.9 vivant',
                                                       'q = 0.1 mort',
                                                       'Max viv sliding',
                                                       'Min mo sliding'])

df_train_vivant_old = df_train_vivant
df_train_mort_old = df_train_mort

for x in xrange:
    
    df_train_vivant[df_train_vivant < x] = 0
    df_train_mort[df_train_mort < x] = 0
    
    max_zero_vi = max_zero(df_train_vivant[df_train_vivant.columns[:-1]])
    max_zero_mo = max_zero(df_train_mort)
    
    slide_max_zero_vi = slide_max_zero(df_train_vivant[df_train_vivant.columns[:-1]])
    slide_max_zero_mo = slide_max_zero(df_train_mort)
    
    df_threshold.loc[x] = [max(max_zero_vi),
                           min(max_zero_mo),
                           np.mean(max_zero_vi),
                           np.mean(max_zero_mo),
                           np.quantile(max_zero_vi,0.9),
                           np.quantile(max_zero_mo,0.1),
                           max(slide_max_zero_vi),
                           min(slide_max_zero_mo)]
    
    
    
#percentage of nonzero values above threshold x (index) 
df_threshold['viv percent'] = count_vi
df_threshold['mort percent'] = count_mo

plt.figure()
plt.plot(xrange,(df_threshold['viv percent'] - df_threshold['mort percent']))


# plot scaled value of difference in other columns. use this as the threshold valu
plt.figure()
plt.plot(xrange,df_threshold['Min mort'] - df_threshold['Max vivant'])

plt.figure()
plt.plot(xrange,df_threshold['mean mort'] - df_threshold['mean vivant'])

plt.figure()
plt.plot(xrange,df_threshold['q = 0.1 mort'] - df_threshold['q = 0.9 vivant'])


# threshold = 17.25
df_train_vivant = df_train_vivant_old
df_train_mort = df_train_mort_old
x = 17.25

df_train_vivant[df_train_vivant < x] = 0
df_train_mort[df_train_mort < x] = 0

#get the max zeroes

#with sliding window this works best with all the values present

# choose threshold as 180 (one hour...) - if it goes beyond this value, it can come back to life, if it does not exceed 500
df_dist = df_dist[animals]
df_alive = pd.DataFrame(index = df_dist.index,columns = df_dist.columns)
df_counters = df_alive.fillna(0)
df_alive = df_alive.fillna(1)

threshold = 250
#create np array of counters of for zero counter (using slider)
    
#counters = dict(zip(df_alive.columns,np.zeros(16)))
counters = np.zeros(16)

for i in range(1,len(df_alive)):
    
    #check if last two contain a zero
    temp = df_dist.iloc[[i-1,i]]
    for col in temp.columns:
        if 0 in temp[col].values:
            counters[col-1] += 1
        else:
            counters[col-1] = 0
            
    df_counters.iloc[i] = counters
    
    
df = df_dist[df_dist.index.day > 14]

# check for once they are dead the longest streak of non zero values

#algorithm to count max nonzero streak - replaces zero with nan, replace positive with zero, replace nan with random
df[df == 0] = np.nan
df[df > 0] = 0
random_data = pd.DataFrame(np.random.randn(df.shape[0],df.shape[1]),index = df.index, columns = df.columns)
df[df.isnull()] = random_data[df.isnull()]

max_nonzero = max_zero(df)
thresh_life = 5 * 3

#counters = np.zeros(16)
counters = dict(zip(df_dist.columns,np.zeros(16)))

for i in range(thresh_life,len(df_alive)):
    
    
    #check if last two contain a zero
    temp = df_dist.iloc[i-thresh_life:i]
    
    #for x,col in enumerate(temp.columns)
    for col in temp.columns:
        #if counters[x] >= 0:
        
        #if alive
        if counters[col] >= 0:
            
            #if 0 in 2 derniers
            if 0 in temp[col].values[-2:]:
                counters[col] += 1
            else:
                counters[col] = 0
        #DEAD        
        else:
            if 0 not in temp[col].values:
                counters[col] = 0
            else: 
                df_alive[col].iloc[i] = 0
                
                
        if counters[col] == threshold:
            counters[col] = -1
            
            
"""
Rework of counting
"""
data = np.array(df_dist)
data_alive = np.ones_like(data)
data_counters = np.zeros_like(data)
counters = np.zeros(16)
alive = np.ones(16)

# through full dataset
for i in range(thresh_life, len(data)):
    
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
            if 0 not in data[(i- thresh_life):i,x]:
                alive[x] = 1
                counters[x] = 0
                
        if counters[x] >= threshold:
            alive[x] = 0
            
        data_alive[i] = alive
        data_counters[i] = counters
        
#replace data with np.nan if dead

df_alive1 = pd.DataFrame(data_alive, index = df_dist.index, columns = df_dist.columns)
df_counters1 = pd.DataFrame(data_counters, index = df_dist.index, columns = df_dist.columns)

# df_dist[df_alive1 == 0] = np.nan
# for col in df_dist.cols: if df_dist.col.is_null() > 20% get rid of column

fig,axe = plot_16(df_alive,colors)
fig_count,axe_count = plot_16(df_counters,colors)
fig1,axe1 = plot_16(df_alive1,colors)
fig_count1,axe_count1 = plot_16(df_counters1,colors)
fig_dist,axe_dist = plot_16(df_dist,colors)


#plot IGT
IGT = []
for i in range(len(df_dist)):
    IGT.append(df_dist.iloc[i][df_alive.iloc[i] == True].min()**2)               