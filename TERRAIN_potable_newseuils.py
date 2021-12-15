# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 14:09:28 2021

find offset seuils

@author: George
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% IMPORT personal mods
os.chdir('MODS')
import dataread_terrain as d_terr
os.chdir('..')

root = r'D:\VP\Viewpoint_data\TERRAIN\SAUR'
os.chdir(root)
files = [f for f in os.listdir() if '.csv' in f]

specie = {'E': 'Erpobdella','G':'Gammarus','R':'Radix'}

def main(files,spec = 'EGR',path = None,start = None,merge = False):
    """
    files
        In file title to decide which file to use
        '{}.csv'.format(container)
        
    specie : str or list str
        G E R
        
    path : Repertoir for data files - necessary for other PCs
    
    start : datetime, optional
        start = pd.to_datetime("01/04/2021 15:00:00", format = "%d/%m/%Y %H:%M:%S")
        The default is None.
    """
    
    if path:
        os.chdir(path)
    else:
        root = r'D:\VP\Viewpoint_data\TERRAIN\Suez'
        os.chdir(root)
    
    print('The following files will be studied:\n',files)
    dfs,dfs_mean = d_terr.read_data_terrain(files,merge,startdate = start,distplot = True)
    
    #%%
    
    #parameters
    seuil_bdf = {'G':[0.7,19],'E':[0.7,18],'R':[0.8,5]}
    # originals cutoff = {'G':[2000,3500,12000],'E':[1000,2500,10000],'R':[250,450,1200]}
    cutoff = {'G':[2000,3500,12000],'E':[1000,2500,10000],'R':[250,450,1200]}
    """
    palier = {
        'E':np.array([2500,5000,7000,30000]),
        'G':np.array([3500,5000,8000,30000]),
        'R':np.array([450,600,950,3000])}
    """
    palier = {
        'E':np.array([cutoff['E'][1],cutoff['E'][1]*2,cutoff['E'][1]*3,cutoff['E'][2]*3],dtype = int),
        'G':np.array([cutoff['G'][1],cutoff['G'][1]*1.5,cutoff['G'][1]*2.25,cutoff['G'][2]*3],dtype = int),
        'R':np.array([cutoff['R'][1],cutoff['R'][1]*1.5,cutoff['R'][1]*2.25,cutoff['R'][2]*3],dtype = int)}
    offsets = d_terr.find_optimum_offsets(palier)
    quant = 0.1
    
    data = {}
    for species in spec:
        df = dfs[species]
        df_mean = dfs_mean[species]
        
        # plot all on same figure - no mean and mean
        d_terr.single_plot(df,species,title = 'Distance covered')
        d_terr.single_plot(df_mean,species,title = 'Distance covered Mean 2min (terrain)')
        
        #calculate mortality and add nan's where animals are dead
        data_alive,data_counters = d_terr.search_dead(np.array(df),species)
        m = np.ones(len(data_alive),dtype = float) - (np.sum(data_alive,axis = 1))/16
        values = np.array(df)
        values[data_alive == 0] = np.nan
        df = pd.DataFrame(data = np.copy(values), index = df.index, columns = df.columns)
        values.sort() # nans moved to back
        
        #match to online algo - 2minute moving mean
        df_mean,m = d_terr.group_meandf(df.copy(), m)
        values = np.array(df_mean)
        values.sort()
        
        #debug
        diff = 15
        for i in range(10):
            #cutoff[species][0] -= diff
            cutoff[species][1] -= diff
            
            palier = {
                'E':np.array([cutoff['E'][1],cutoff['E'][1]*2,cutoff['E'][1]*3,cutoff['E'][2]*3],dtype = int),
                'G':np.array([cutoff['G'][1],cutoff['G'][1]*1.5,cutoff['G'][1]*2.25,cutoff['G'][2]*3],dtype = int),
                'R':np.array([cutoff['R'][1],cutoff['R'][1]*1.5,cutoff['R'][1]*2.25,cutoff['R'][2]*3],dtype = int)}
            offsets = d_terr.find_optimum_offsets(palier)
        
            thresholds_percent = [seuil_bdf,cutoff,offsets,quant]
            
            IGTper_mean,old_IGT = d_terr.IGT_percent_and_old(values,species,np.zeros_like(m),np.zeros_like(m),thresholds_percent)
            fig,axe = d_terr.double_vertical_plot(old_IGT,IGTper_mean,ind = df_mean.index)
            d_terr.add_mortality(fig,axe,m,ind = df_mean.index)
        
        # #calculate moving mean
        # IGT_mean = np.array(pd.Series(IGTper).rolling(8).mean())
        
        data.update({species:{'df':df,'df_m':df_mean,'mort':m,'IGT':IGTper_mean}})
        
    return data

main(files,spec = 'R',path = r'D:\VP\Viewpoint_data\TERRAIN\SAUR')

"""
Findings

Gammarus - 1000 off works ok with 10 quantile. red 2000, orange 1500, quantile 10%  - diff set to 100
Erpobdella - Max : drop top lim to 1500, Min : Just change quantile to 10%     - diff set to 100
Radix - Descendre 125 - 325, quantile 10%      - diff set to 15
"""