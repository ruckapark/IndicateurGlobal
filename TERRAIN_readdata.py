# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 14:09:28 2021

With downloaded data files; analyse terrain data files.

@author: George
"""

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

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

os.chdir('MODS')
import dataread_terrain as d_terr
os.chdir('..')

def main(files,spe,root = None,potable = False):
    
    """ Main of code below """
    if root == None: root = r'D:\VP\Viewpoint_data\TERRAIN\Suez'
    os.chdir(root)
    
    #more than one file must be consecutive files
    if len(files) == 1:
        merge = False
    else:
        merge = True
        
    specie = {'E': 'Erpobdella','G':'Gammarus','R':'Radix'}
    
    print('The following files will be studied:\n',files)
    dfs,dfs_mean = d_terr.read_data_terrain(files,merge)
    
    #seuils for toxicity calculation
    thresholds_percent = d_terr.return_IGT_thresh(potable)
    
    data = {}
    for species in spe:
        df = dfs[species]
        df_mean = dfs_mean[species]
        
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
        
        IGTper_mean,old_IGT = d_terr.IGT_percent_and_old(values,species,np.zeros_like(m),np.zeros_like(m),thresholds_percent)
        fig,axe = d_terr.double_vertical_plot(old_IGT,IGTper_mean,ind = df_mean.index)
        d_terr.add_mortality(fig,axe,m,ind = df_mean.index)
        
        data.update({species:{'df':df,'df_m':df_mean,'mort':m,'IGT':IGTper_mean}})
    
    return data,thresholds_percent

#%% IMPORT personal mods
if __name__ == "__main__":
    
    root = r'D:\VP\Viewpoint_data\TERRAIN\SAUR'
    os.chdir(root)
    files = [f for f in os.listdir() if '.csv' in f]
    files = d_terr.sort_filedates(files)
    files = [files[0]]
    merge = True
    
    specie = {'E': 'Erpobdella','G':'Gammarus','R':'Radix'}
    
    print('The following files will be studied:\n',files)
    dfs,dfs_mean = d_terr.read_data_terrain(files,merge,startdate = None,distplot = False)
    
    #parameters and seuils for calculations depending on potable or not
    potable = True
    thresholds_percent = d_terr.return_IGT_thresh(potable)
    
    #%%
    data = {}
    spec = 'EGR'
    for species in spec:
        df = dfs[species]
        df_mean = dfs_mean[species]
        
        # plot all on same figure - no mean and mean
        #d_terr.single_plot(df_mean,species,title = 'Distance covered Mean 2min (terrain)')
        
        # plot individually (plot16)
        # fig,axe = d_terr.plot_16(df,title = specie[species])
        # fig,axe = d_terr.plot_16(df_mean,title = 'Mean {}'.format(specie[species]))
        
        #calculate mortality and add nan's where animals are dead
        data_alive,data_counters = d_terr.search_dead(np.array(df),species)
        m = np.ones(len(data_alive),dtype = float) - (np.sum(data_alive,axis = 1))/16
        values = np.array(df)
        values[data_alive == 0] = np.nan
        df = pd.DataFrame(data = np.copy(values), index = df.index, columns = df.columns)
        values.sort() # nans moved to back
        
        #compare old and new values (IGTper = IGT percentage 0 - 100% toxicité)
        #IGTper,old_IGT = d_terr.IGT_percent_and_old(values,species,np.zeros_like(m),np.zeros_like(m),thresholds_percent)
        #fig,axe = d_terr.double_vertical_plot(old_IGT,IGTper,ind = df.index)
        
        #match to online algo - 2minute moving mean
        df_mean,m = d_terr.group_meandf(df.copy(), m)
        values = np.array(df_mean)
        values.sort()
        
        
        IGTper_mean,old_IGT = d_terr.IGT_percent_and_old(values,species,np.zeros_like(m),np.zeros_like(m),thresholds_percent)
        fig,axe = d_terr.double_vertical_plot(old_IGT,IGTper_mean,ind = df_mean.index)
        d_terr.add_mortality(fig,axe,m,ind = df_mean.index)
        
        data.update({species:{'df':df,'df_m':df_mean,'mort':m,'IGT':IGTper_mean}})