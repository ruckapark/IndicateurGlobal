# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 11:08:57 2022

@author: George
"""

#global variables
"""
Possibilities
copper_entries = ['763_Copper5.xls',
                '764_Copper4.xls',
                '765_Copper5.xls',
                '765_Copper6.xls',
                '767_Copper3.xls',
                '767_Copper7.xls',
                '767_Copper9.xls',
                '767_Copper10.xls']

meth_entries = ['760_Methomyl3.xls',
                '761_Methomyl2.xls',
                '761_Methomyl3.xls',
                '762_Methomyl1.xls',
                '769_Methomyl1.xls'
                ]
"""

copper_entries = ['763_Copper5.xls',
                  '764_Copper4.xls',
                  '765_Copper5.xls',
                  '767_Copper7.xls',
                  '767_Copper9.xls']

meth_entries = ['760_Methomyl4.xls',
                '761_Methomyl1.xls',
                '761_Methomyl3.xls',
                '761_Methomyl5.xls',
                '769_Methomyl1.xls'
                ]

def index_hours(index,dopage):
    #return hours as float 0 = dopage
    hours = (index - dopage).total_seconds()/3600
    return hours

#%% IMPORTS classic mods
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn import preprocessing
from datetime import timedelta

#%% IMPORT personal mods
os.chdir('MODS')
from data_merge import merge_dfs
from dope_reg import dope_read
#import gen_pdf as pdf
import dataread as d_
os.chdir('..')

specie = {'E': 'Erpobdella','G':'Gammarus','R':'Radix'}

def find_reaction(df,dopage = 0):
    if dopage:
        delta = pd.Timedelta(hours = 1)
    else:
        delta = 1
    reaction_period = df[df.index > dopage + delta]
    try:
        return reaction_period[reaction_period > 100].index[0]
    except:
        return None
    
def block_mean(df,timestep = 2,unit = 'm'):
    
    """
    Coded for timestep block mean and not integer
    """
    
    df = df.copy()
    df['t'] = (60*(df.index - df.index[0]))//timestep
    df_m = df.groupby('t').mean()
    df_m.index = df.index[0] + timestep/60
    return df_m

#%% code
if __name__ == "__main__":
    
    #loop directories
    substances = ['meth','copper']
    directories = {'meth':r'D:\VP\ARTICLE1_methomyl\Data','copper':r'D:\VP\ARTICLE1_copper\Data'}
    files_reg = {'meth':meth_entries,'copper':copper_entries}
    
    #Article1 data
    for sub in substances:
        
        #locate dir
        os.chdir(directories[sub])
        
        #compressed files have " _ " in file name
        files = files_reg[sub]
        dope_df = dope_read('{}_reg'.format(sub))
        
        #treat each data file, add to dictionary
        data = {}
        dopages = {}
        for file in files:
            
            #Toxname
            Tox = int(file[:3])
            
            #extract only gammarus df
            df = d_.read_merge([file])
            dfs = d_.preproc(df)
            df = dfs['G']
            deadcol = d_.remove_dead(df,'G')
            if len(deadcol) > 14: continue
            df = df.drop(columns = deadcol)
            
            #meaned df
            t_mins = 2
            #df_mean = d_.rolling_mean(df,t_mins)
            df_mean = d_.block_mean(df,t_mins)
            df_mean = d_.rolling_mean(df_mean,5)
            
            #find dopage time
            dopage,date_range,conc,subs,molecule,etude_ = d_.dope_params(dope_df,Tox,df.index[0],df.index[-1])
            
            data.update({file:[df.set_index(index_hours(df.index, dopage)),df_mean.set_index(index_hours(df_mean.index, dopage))]})
            #dopages.update({file:[dopage,date_range,conc]})
            
        #%% analysis
        sns.set_style("white")
        fig = plt.figure(figsize = (8,6))
        axe = fig.add_axes([0.15,0.1,0.8,0.8])
        axe.axvline(color = 'red')          #dopage at zero
        # palet = ['#0b5394','#126375','#6aa84f','#38761d','#274e13']
        palet = sns.color_palette()[:5]
        
        for i,study in enumerate(data):
            
            #data
            [df,df_mean] = data[study]
            
            mean_dist = df_mean.mean(axis = 1)
            quantile_dist = df_mean.quantile(q = 0.05, axis = 1)**2
            
            #fig,axe = d_.single_plot(mean_dist,title = 'Mean : {}'.format(conc))
            #d_.dataplot_mark_dopage(axe,date_range)
            
            if sub == 'meth':
                axe.plot(quantile_dist,color = palet[i],label = 'Study {}'.format(i+1))
                start_reaction = find_reaction(quantile_dist)
                # if start_reaction:
                #     axe.axvline(start_reaction,color = 'orange')
            else:
                axe.plot(quantile_dist[quantile_dist.index >= -1],color = palet[i],label = 'Study {}'.format(i+1))
            
        axe.set_xticklabels(np.array(axe.get_xticks(),dtype = np.int64),fontsize = 14)
        axe.set_yticklabels(np.array(axe.get_yticks(),dtype = np.int64),fontsize = 14)
        plt.tight_layout()
        plt.title(sub + ' spike $100\mu gL^{-1}$', fontsize = 18)
        plt.ylabel('Periodic quantile distance $(mm^{2}\cdot20^{-1}$)', fontsize = 16)
        plt.xlabel('Spike obersvation time $(hours)$', fontsize = 16)
        plt.legend(fontsize = 16)
        sns.despine()
        
        #fig.savefig(r'C:\Users\Admin\Documents\Viewpoint\Article1\{}_{}'.format('Fig2A',sub))