# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 10:48:47 2022

@author: George
"""


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

os.chdir('MODS')
import dataread_terrain as d_terr
os.chdir('..')

def main(files,spe,root = None,potable = False,
         thresholds = {'G':190,'E':180,'R':50}):
    
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
    dfs,dfs_mean = d_terr.read_data_terrain(files,merge,thresholds = thresholds)
    
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
        values.sort() #nans moved to back
        
        #match to online algo - 2minute moving mean
        df_mean,m = d_terr.group_meandf(df.copy(), m)
        values = np.array(df_mean)
        values.sort()
        
        IGTper_mean,old_IGT = d_terr.IGT_percent_and_old(values,species,np.zeros_like(m),np.zeros_like(m),thresholds_percent)
        fig,axe = d_terr.double_vertical_plot(old_IGT,IGTper_mean,ind = df_mean.index)
        d_terr.add_mortality(fig,axe,m,ind = df_mean.index)
        
        data.update({species:{'df':df,'df_m':df_mean,'mort':m,'IGT':IGTper_mean}})
    
    return data,thresholds_percent

def signal_bio(df,sp_main = 'G'):
    """
    Entry is dataframe (df) with 3 columns - Erpobdella, Gammarus, Radix
    
    Creat column 'Case' with conditions defined
    
    Add column Signal biologique
    """
    
    #calculate values based on case column
    df['case'] = np.where(df[[*specie]].max(axis = 1) < 50, 0,
                    np.where(((df[[*specie]].max(axis = 1) > 50) & (df[[*specie]].max(axis = 1) < 75)) & (df[[*specie]].median(axis = 1) < 50), 1,
                    np.where((df[[*specie]].max(axis = 1) > 75) & (df[[*specie]].median(axis = 1) < 50), 2,
                    np.where(((df[[*specie]].max(axis = 1) > 50) & (df[[*specie]].max(axis = 1) < 75)) & (df[[*specie]].median(axis = 1) > 50), 3,
                    np.where(df[[*specie]].median(axis = 1) > 75, 4, np.nan)))))
    
    #Based on column work out signal bio value - 
    signalbio = np.ones(df.shape[0])
    cases = np.array(df['case'])
    prio = [*specie].index(sp_main) #locate priority species
    for i in range(df.shape[0]):
        case = cases[i]
        igts = df[[*specie]].iloc[i]
        
        #Case 0 : no alert - default priority species (prio - Gammarus)
        if case == 0:
            if igts[prio] < 10:
                signalbio[i] = igts[prio]
            else:
                signalbio[i] = igts.min()
                
        #Case 1 : medium alert - one specie - mean of three
        elif case == 1:
            signalbio[i] = igts.mean()
            
        #Case 2 : high alert - one species - greater than 75
        elif case == 2:
            #check recent history - 2 minute timestep
            sp = igts.idxmax() #determine species of max value
            if i == 0:
                igt_recent = df[sp].iloc[i]
            elif i <= 10:
                igt_recent = df[sp].iloc[:i]
            else:
                igt_recent = df[sp].iloc[i-10:i]
            
            #if recent spike
            if igt_recent.max() > 50:
                signalbio[i] = igts.max()
            #elif high average value
            elif (igts.min() + igts.median())/2 > 30:
                signalbio[i] = igts.max()
            #else reclibrate to medium warning
            else:
                signalbio[i] = igts.max() - 15
            
        #Case 4 : medium alert - multiple species - mid value
        elif case == 3:
            
            #if all high alert critical warning
            if igts.mean() > 50:
                signalbio[i] = igts.mean() + 15
            else:
                signalbio[i] = igts.max()
        
        #Case 5 : high alert - multiple species
        elif case == 4:
            
            signalbio[i] = igts.max()
        
        #Should never happen - error case
        else:
            signalbio[i] = np.nan
    
    df['bio'] = signalbio
    return df

#%% IMPORT personal mods
if __name__ == "__main__":
    
    root = r'D:\VP\Viewpoint_data\TERRAIN\Suez'
    os.chdir(root)
    files = [f for f in os.listdir() if '.csv' in f]
    #files = d_terr.sort_filedates(files)
    files = [files[1]]
    merge = True
    
    specie = {'E': 'Erpobdella','G':'Gammarus','R':'Radix'}
    
    print('The following files will be studied:\n',files)
    dfs,dfs_mean = d_terr.read_data_terrain(files,merge,startdate = None,distplot = False)
    
    #parameters and seuils for calculations depending on potable or not
    potable = False
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
        #d_terr.add_mortality(fig,axe,m,ind = df_mean.index)
        
        data.update({species:{'df':df,'df_m':df_mean,'mort':m,'IGT':IGTper_mean}})

    #main(['toxmate_051121-191121.csv'],'EGR',root = r'D:\VP\Viewpoint_data\TERRAIN\AltenRhein774')
    
    #%%test of signal bio
    IGT = pd.DataFrame({'E':data['E']['IGT'],'G':data['G']['IGT'],'R':data['R']['IGT']}) #BUG if IGT arrays different lenght ?!
    IGT = signal_bio(IGT)
    
    #visualise with gammarus
    fig,axe = d_terr.double_vertical_plot(IGT['bio'],IGT['G'],ind = df_mean.index)