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

#Plotting graphics
import matplotlib.pyplot as plt
import seaborn as sns #style defined in dataread

#system
import os
import pandas as pd
import numpy as np
import scipy as sp

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
    
    run = 'control'
    #run = 'spike'
    
    if run == 'control':
        Tox,species,etude = 766,'G',37 #temoin
    else:
        Tox,species,etude = 767,'G',36 #spike
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
    
    #convert to numeric
    exp_start = df.index[0] - pd.Timedelta(hours = 20)
    dopage = (dopage - exp_start).total_seconds()/3600
    
    #apply calibration scale
    scale = 4
    df = df/4
    
    #apply threshold
    df = df.where(df < 200, 0)
    
    #%%
    
    # apply moving mean
    #t = 15 #ten minutes for clarity
    #df_roll = d_.rolling_mean(df, t)
    #df_mean = d_.block_mean(df_roll,t)
    df_mean = d_.rolling_mean(d_.block_mean(df,10), 4)
    
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
    #start date (use 17h on the monday before dopage)
    df_mean.index = (df_mean.index - exp_start).total_seconds()/3600
    df_mean.index = (df_mean.index - dopage)
    df_mean = df_mean[(df_mean.index > -15) & (df_mean.index < 22)]
    
    dopage = 0
    
    #regression fit
    medians = df_mean.median(axis = 1)
    
    #%% inspired by single plot 16
    if run == 'control':
        fig = plt.figure(figsize = (11,3.5))
    else:
        fig = plt.figure(figsize = (11,5))
    
    with sns.axes_style("white"):
        
        #for control figure
        if run == 'control':
            axe = fig.add_axes([0.13,0.2,0.77,0.7])
            axe2 = axe.twinx()
            axe.tick_params(axis = 'x', rotation = 0)
            axe.set_title('') #control
            axe.fill_between(df_mean.index,df_mean.quantile(0.25,axis = 1),df_mean.quantile(0.75,axis = 1),color = '#1492c4',alpha = 0.75,zorder = 2)
            axe.plot(df_mean.median(axis = 1),'#2d04c2')
            
            axe.axvline(dopage - 0.25, color = 'black', linestyle = '--',linewidth = 2)
            
            axe.set_ylim((0,60))
            axe2.set_ylim((0,300))
            axe2.plot(df_mean.quantile(0.1,axis = 1)**2,'red',zorder = 2,label = 'Squared quantile Q(0.05)',linewidth = 2.5)
            
            axe.set_yticks(np.linspace(0, axe.get_ybound()[1], 3))
            axe2.set_yticks(np.linspace(0, axe2.get_ybound()[1], 3))
            
            
            axe.set_xlabel('')
            axe.set_ylabel('')
            axe2.set_ylabel('')
            
            #axe.legend(fontsize = 17)
            axe.annotate('Control', xy=(18,50),
                         size = 17,
                         bbox = dict(boxstyle = 'round', fc = '#d4d2d2'))
            
            axe.set_xticklabels(np.array(axe.get_xticks(),dtype = np.int64),fontsize = 14)
            axe.set_yticklabels(np.array(axe.get_yticks(),dtype = np.int64),fontsize = 14)
            axe2.set_yticklabels(np.array(axe2.get_yticks(),dtype = np.int64),fontsize = 14)
            
            plt.tight_layout()
            
            fig.savefig(r'C:\Users\George\Documents\{}'.format('Fig1Bcon')) #PC
            #fig.savefig(r'C:\Users\Admin\Documents\Viewpoint\Figures\{}'.format('Fig1Bcon')) #Laptop
            
            
        #for dopage with cu
        else:
            axe = fig.add_axes([0.13,0.13,0.77,0.77])
            axe2 = axe.twinx()
            axe.tick_params(axis = 'x', rotation = 0)
            #axe.set_title('Activity Distribution and Avoidance Signal for $100ugL^{-1}$ Copper spike', fontsize = 22) #spike
            axe.fill_between(df_mean.index,df_mean.quantile(0.25,axis = 1),df_mean.quantile(0.75,axis = 1),color = '#1492c4',alpha = 0.75,zorder = 2,label = 'Interquartile range')
            axe.plot(df_mean.median(axis = 1),'#2d04c2',label = 'Median')
            
            axe.axvline(dopage - 0.25, color = 'black', linestyle = '--',linewidth = 2, label = 'Spike')
            
            axe.set_ylim((0,120))
            axe2.set_ylim((0,500))
            axe2.plot(df_mean.quantile(0.1,axis = 1)**2,'red',zorder = 2,linewidth = 2.5)
            
            axe.set_yticks(np.linspace(0, axe.get_ybound()[1], 5))
            axe2.set_yticks(np.linspace(0, axe2.get_ybound()[1], 5))
            
            axe.legend(loc = 2,fontsize = 18)
            #axe2.legend(fontsize = 17)
            
            axe.annotate('$100ugL^{-1}$ copper', xy=(12.5,105),
                         size = 17,
                         bbox = dict(boxstyle = 'round', fc = '#d4d2d2'))
        
        
        
            axe.set_xlabel('Observation time $(hours)$',fontsize = 20)
            axe.set_ylabel('Distance $(mm\cdot20s^{-1}$)',fontsize = 20,color = 'blue')
            axe2.set_ylabel('Squared Lower Quantile $(mm^{2}\cdot20s^{-1})$',fontsize = 18,color = 'red')
            
            axe.set_xticklabels(np.array(axe.get_xticks(),dtype = np.int64),fontsize = 14)
            axe.set_yticklabels(np.array(axe.get_yticks(),dtype = np.int64),fontsize = 14)
            axe2.set_yticklabels(np.array(axe2.get_yticks(),dtype = np.int64),fontsize = 14)
            
            plt.tight_layout()
            
            #fig.savefig(r'C:\Users\George\Documents\{}'.format('Fig1B')) #PC
            #fig.savefig(r'C:\Users\Admin\Documents\Viewpoint\Figures\{}'.format('Fig1B')) #Laptop