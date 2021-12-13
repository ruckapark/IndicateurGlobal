# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 10:10:12 2021

File created for the weekly run of visualising data

@author: Admin
"""

#%% IMPORTS

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from datetime import timedelta

#%% IMPORT personal mods
os.chdir('MODS')
from data_merge import merge_dfs
from dope_reg import dope_read
import dataread as d_
os.chdir('..')

dopages = {
    'flumequin_5':'2018-06-08 13:52:00',
    'flumequine':'2018-06-08 13:51:00',
    'diclofenac':'2018-04-27 13:37:30'}

#%% FUNCTIONS - plotting functions could also be put into another module         

if __name__ == '__main__':
                
    species = 'G'
    specie = {'E': 'Erpobdella','G':'Gammarus','R':'Radix'}
    
    root = r'D:\VP\Data_alex\ToxMate\calibration_2018\1-Donnees\flumequine\flumequine'
    os.chdir(root)
    #files = [file for file in os.listdir() if os.path.isfile(file)]
    files = ['flumequin_5.csv']
    sub = files[0].split('.')[0]
    Tox = 'OLD'
    etude = 00
     
    df = d_.read_merge(files,oldfiles = True)
    dfs = d_.preproc(df,oldfiles = True)
    
    # could also seperate out over time
    df = dfs[species]
    df = df.drop(columns = d_.remove_dead(df,species))
    
    
    #%% plot
    
    #create dope reg for alex's experiment
    #dope_df = dope_read('Alex-reg.csv')
    #dopage,date_range,conc,sub,molecule,etude = d_.dope_params(dope_df,Tox,df.index[0],df.index[-1])
    dopage = pd.to_datetime(dopages[sub], format = '%Y-%m-%d %H:%M:%S')
    
    fig,axe = d_.single_plot16(df, species, title = '20s distance - {}'.format(specie[species]))
    #d_.dataplot_mark_dopage(axe,date_range)
    
    # moving means
    t_mins = 5
    df_mean = d_.rolling_mean(df,t_mins)
    fig,axe = d_.single_plot16(df_mean, species, title = '{} Moving mean - {}'.format(t_mins,specie[species]))
    #d_.dataplot_mark_dopage(axe,date_range)
    
    #fig,axe = d_.plot_16(df_mean[(df_mean.index > (dopage - pd.Timedelta(hours = 3)))&(df_mean.index < (dopage + pd.Timedelta(hours = 12)))],mark = date_range)
    #fig,axe = d_.plot_16(df_mean[(df_mean.index > (dopage - pd.Timedelta(hours = 1)))&(df_mean.index < (dopage + pd.Timedelta(hours = 2)))],mark = date_range)
    
    """
    population mean
    IGT
    """
    
    #plot all the means across cells
    mean_dist = df_mean.mean(axis = 1)
    quantile_dist = df_mean.quantile(q = 0.05, axis = 1)**2
    
    fig,axe = d_.single_plot(mean_dist,title = 'Mean distance')
    fig,axe = d_.single_plot(quantile_dist,title = 'IGT')
    #d_.dataplot_mark_dopage(axe,date_range)
    
    fig,axe = d_.single_plot(mean_dist[(mean_dist.index > (dopage - pd.Timedelta(hours = 3)))&(mean_dist.index < (dopage + pd.Timedelta(hours = 9)))],title = 'Mean distance')
    #d_.dataplot_mark_dopage(axe,date_range)
    
    #title_ = 'ToxIndex: {}({}), {}   {}-{}'.format(sub,molecule,conc,date_range[0].strftime('%d/%m'),date_range[1].strftime('%d/%m'))
    
    fig,axe = d_.single_plot(quantile_dist[(quantile_dist.index > dopage - pd.Timedelta(hours = 18)) & (quantile_dist.index < dopage + pd.Timedelta(hours = 36))])
    #d_.dataplot_mark_dopage(axe,date_range)
    
    #smaller plot
    IGT = quantile_dist[(quantile_dist.index > dopage - pd.Timedelta(minutes = 60)) & (quantile_dist.index < dopage + pd.Timedelta(minutes = 120))]
    MEANS = mean_dist[(mean_dist.index > dopage - pd.Timedelta(hours = 3)) & (mean_dist.index < dopage + pd.Timedelta(hours = 9))]
    fig,axe = d_.single_plot(IGT, title = 'short')
    #axe.set_xlabel('Tox Ind')
    #d_.dataplot_mark_dopage(axe,date_range)
    
    results_m = pd.DataFrame({'MEANS_{}_{}_{}'.format(sub,Tox,etude):np.array(MEANS)})
    results_IGT = pd.DataFrame({'IGT_{}_{}_{}'.format(sub,Tox,etude):np.array(IGT)})
    reg_means = pd.read_csv(r'C:\Users\George\Documents\SETAC\DATA_means.csv')
    reg_IGT = pd.read_csv(r'C:\Users\George\Documents\SETAC\DATA_IGT.csv')
    if results_m.columns[0] not in reg_means.columns:
        reg_means = pd.concat([reg_means,results_m],axis = 1)
        reg_IGT = pd.concat([reg_IGT,results_IGT],axis = 1)
    reg_means.to_csv(r'C:\Users\George\Documents\SETAC\DATA_means.csv',index = False)
    reg_IGT.to_csv(r'C:\Users\George\Documents\SETAC\DATA_IGT.csv', index = False)