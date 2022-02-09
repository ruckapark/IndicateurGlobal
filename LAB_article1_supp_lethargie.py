# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 10:03:48 2022

Supplementary data - show the descente to lethargie for multiple figure
All pre dopage

@author: Admin
"""

#Plotting graphics
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
sns.set_style('ticks')

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


def single_plot(df,axe,species,title = '',ticks = None,colors = None):
    for i in df.columns:
        axe.plot(df.index,df[i],label = '{}{}'.format(species,i),color = colors[i-1],zorder = 1)


if __name__ == "__main__":
    
    study = {33:[760,763,765,766,767,768],
             36:[761,763,764,765,766,767],
             37:[762,763,764,765,766,768]}
    
    for etude in study:
    
        Toxs = study[etude]
        species = 'G'
        specie = {'E': 'Erpobdella','G':'Gammarus','R':'Radix'}
        
        fig,axes = plt.subplots(2,3,sharex = True,sharey = True,figsize = (20,8))
        plt.suptitle('Etude {} Distance - {}'.format(study,specie[species]))
        
        for i,Tox in enumerate(Toxs):
            os.chdir(r'D:\VP\Viewpoint_data\TxM{}-PC\{}'.format(Tox,d_.study_no(etude)))
            files = [file for file in os.listdir() if os.path.isfile(file)]
            
            if len(os.listdir()) == 2:
                df = d_.read_merge(files)
                dfs = d_.preproc(df)
            else:
                continue
            
            df = dfs[species]
            df = df.drop(columns = d_.remove_dead(df,species))
            
            dope_df = dope_read()
            dopage,date_range,conc,sub,molecule,etude_ = d_.dope_params(dope_df,Tox,df.index[0],df.index[-1])
            df = df[df.index < dopage]
            
            #apply calibration scale
            scale = 4
            df = df/4
            
            #apply threshold
            df = df.where(df < 200, 0)
            
            #%%
            
            # apply moving mean
            t = 20 #ten minutes for clarity
            df_mean = d_.block_mean(df,t)
            df_roll = df_mean.rolling(15).mean()
            
            c = 16*['#a8a8a8']
            
            axe = axes[i//3,i%3]
            single_plot(df_mean, axe, species,colors = c)
            axe.fill_between(df_roll.index,df_roll.quantile(0.35,axis = 1),df_roll.quantile(0.65,axis = 1),facecolor = 'cornflowerblue',zorder = 2)
            axe.plot(df_roll.quantile(0.35,axis = 1),color = 'royalblue',linestyle = 'dotted')
            axe.plot(df_roll.quantile(0.65,axis = 1),color = 'royalblue',linestyle = 'dotted')
            axe.plot(df_roll.median(axis = 1),'blue',lw = 3.0)