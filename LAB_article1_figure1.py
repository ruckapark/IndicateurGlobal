# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 16:17:27 2022

Article 1 - Figure 1

Present for Gammarus a long dataset showing :
    - each trajectory over 5 minute
    - mean trajectory
    - IGT
    - Lethargie (like plot with regression)

@author: Admin
"""

"""
Details of dataset:
    - 01-07/06 TXM763 - (Manganese) perhaps best to be copper
    - 
"""

#Plotting graphics
import matplotlib.pyplot as plt
import seaborn as sns #style defined in dataread

#system
import os
import pandas as pd
import numpy as np

#parent files
os.chdir('MODS')
from data_merge import merge_dfs
from dope_reg import dope_read
import dataread as d_
os.chdir('..')

def index_hours(index,dopage):
    #return hours as float 0 = dopage
    hours = (index - dopage)/3600
    return hours

if __name__ == "__main__":
    
    Tox,species,etude = 763,'G',33
    specie = {'E': 'Erpobdella','G':'Gammarus','R':'Radix'}
    
    os.chdir(r'D:\VP\Viewpoint_data\TxM{}-PC\{}'.format(Tox,d_.study_no(etude)))
    files = [file for file in os.listdir() if os.path.isfile(file)]
    
    df = d_.read_merge(files)
    dfs = d_.preproc(df)
    
    df = dfs[species]
    df = df.drop(columns = d_.remove_dead(df,species))
    
    dope_df = dope_read()
    dopage,date_range,conc,sub,molecule,etude_ = d_.dope_params(dope_df,Tox,df.index[0],df.index[-1])
    df = df[df.index < dopage + pd.Timedelta(1,'d')]
    
    #apply calibration scale
    scale = 4
    df = df/4
    
    #apply threshold
    df = df.where(df < 200, 0)
    
    #%%
    
    # apply moving mean
    t = 30 #ten minutes for clarity
    df_roll = d_.rolling_mean(df, t)
    df_mean = d_.block_mean(df,t)
    
    #add to dataread a block mean
    
    #add new col inside function with a timegroup
    
    #use groupby function and mean
    
    """
    test
    
    def block_mean(df,step,time = True):
        if time:
            df['index_column'] = df.index
    
    """
    
    #custom colors
    c = '#7a7a7a'
    """
    c = [
    '#42f5e0','#12aefc','#1612fc','#6a00a3',
    '#8ef743','#3c8f01','#0a4001','#fc03ca',
    '#d9d200','#d96c00','#942c00','#fc2803',
    '#e089b6','#a3a3a3','#7a7a7a','#303030'
    ]
    """
    
    #inspired by single plot 16
    fig = plt.figure(figsize = (13,8))
    with sns.axes_style("white"):
        axe = fig.add_axes([0.1,0.1,0.8,0.8])
        axe2 = axe.twinx()
        # for i in df_mean.columns:
        #     axe.plot(df_mean.index,df_mean[i],label = '{}{}'.format(species,i),color = '#7a7a7a',zorder = 1)
        axe.tick_params(axis = 'x', rotation = 90)
        axe.set_title('Plot 1')
        
        axe.plot(df_mean.median(axis = 1),'blue')
        axe.plot(df_mean.quantile(0.05,axis = 1),'blue',linestyle= (0, (5, 10)),zorder = 2)
        axe.plot(df_mean.quantile(0.95,axis = 1),'blue',linestyle= (0, (5, 10)),zorder = 2)
        axe.fill_between(df_mean.index,df_mean.quantile(0.25,axis = 1),df_mean.quantile(0.75,axis = 1),color = '#1492c4',alpha = 0.75,zorder = 2)
        
        #axe2.plot(df_mean.quantile(0.05,axis = 1)**2,'red',linewidth = 2,zorder = 1)
        
        axe.axvspan(date_range[0] - pd.Timedelta(minutes = 35), date_range[1] - pd.Timedelta(minutes = 35), alpha=0.7, color='orange')
        axe.axvline(dopage - pd.Timedelta(minutes = 35), color = 'black', linestyle = '--',linewidth = 2)
        
        axe.set_ylim((0,140))
        axe2.set_ylim((0,60))
        
        axe.set_yticks(np.linspace(0, axe.get_ybound()[1], 5))
        axe2.set_yticks(np.linspace(0, axe2.get_ybound()[1], 5))