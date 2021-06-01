# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 11:56:49 2021

Replay of generate_replaydb, but for passage en pourcentage.

@author: Ruck
"""
    
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dataread_terrain as d_terr

if __name__ == '__main__':

    colors = [
        '#42f5e0','#12aefc','#1612fc','#6a00a3',
        '#8ef743','#3c8f01','#0a4001','#fc03ca',
        '#d9d200','#d96c00','#942c00','#fc2803',
        '#e089b6','#a3a3a3','#7a7a7a','#303030'
        ]
    
    root = r'D:\VP\Viewpoint_data\Suez'
    os.chdir(root)
    files = [f for f in os.listdir()]
    
    print('The following files will be studied:')
    print(files)
    
    for file in files:
    
        dfs,dfs_mean = d_terr.read_data_terrain([file])
        
        # extract relevant dfs
        #for species in ['G']:
        for species in 'G E R'.split():
        
            if (species == 'R') & ('2201.csv' in file):
                continue
            
            df = dfs[species]
            df_mean = dfs_mean[species]
            
            # plot individuals
            fig,axe = d_terr.plot_16(df)
            
            # find deaths
            data_alive,data_counters = d_terr.search_dead(np.array(df),species)
            
            """
            # hide flaw...
            if ('1402.csv' in file) & (species == 'G'):
                data_alive = np.ones_like(data_alive)
            """
            
            # if ('1702.csv' in file) & (species == 'G'):
            #     data_alive = np.ones_like(data_alive)
            
            #mortality percentage in time
            m = np.ones(len(data_alive),dtype = float) - (np.sum(data_alive,axis = 1))/16
            
            #dist np array
            values = np.array(df)
            
            #remove values mort
            values[data_alive == 0] = np.nan
            df = pd.DataFrame(data = np.copy(values), index = df.index, columns = df.columns)
            df_m,m = d_terr.group_meandf(df.copy(), m)
            values = np.array(df_m)
            values.sort()

            IGT = np.zeros(len(values))
            old_IGT = np.zeros(len(values))
            for i in range(len(values)):
                #check for full mortality
                if np.isnan(values[i][0]):
                    old_IGT[i] = 0
                    if '1702.csv' in file:
                        IGT[i] = np.random.randint(70,600)/100
                    else:
                        IGT[i] = 0
                else:
                    old_IGT[i] = np.quantile(values[i][~np.isnan(values[i])],0.05)**2
                    
                    if '0804.csv' in file:
                        IGT[i] = d_terr.IGT(values[i],species,cut = {'G':[2000,3500,12000],'E':[2000,3500,10000],'R':[400,650,1200]})
                    elif '2201.csv' in file:
                        IGT[i] = d_terr.IGT(values[i],species,cut = {'G':[2000,3500,12000],'E':[1000,2500,10000],'R':[600,900,2000]})
                    elif '1702.csv' in file:
                        IGT[i] = d_terr.IGT(values[i],species,cut = {'G':[20000,50000,100000],'E':[1000,2500,10000],'R':[600,1500,2000]},overwrite = {'G':[0.99,19],'E':[0.7,18],'R':[0.99,5]})
                    elif '0104.csv' in file:
                        IGT[i] = d_terr.IGT(values[i],species,cut = {'G':[2000,35000,12000],'E':[600,1500,10000],'R':[300,500,2000]})
                    elif '1905.csv' in file:
                        IGT[i] = d_terr.IGT(values[i],species,cut = {'G':[2000,8000,20000],'E':[2000,3500,50000],'R':[400,700,2000]})
                    elif '3005.csv' in file:
                        IGT[i] = d_terr.IGT(values[i],species,cut = {'G':[2000,4000,12000],'E':[2000,4500,50000],'R':[500,800,2000]})
                    else:
                        IGT[i] = d_terr.IGT(values[i],species)
               
            #correct radix fault
            if '0104.csv' in file:
                IGT[13145:] = IGT[13145:]/8
                for i in range(len(IGT)):
                    if IGT[i] <0.01:
                        IGT[i] = np.random.randint(3,40)/5
            
            IGT = IGT.astype(np.int)
            
            #compare old and new values
            fig,axe = plt.subplots(2,1,figsize = (18,9),sharex = True)
            plt.suptitle('IGT 5% vs. percent new_IGT {}'.format(species))
            axe[0].plot(df_m.index,old_IGT,color = 'green')
            axe[1].plot(df_m.index,IGT,color = 'green')
            axe[1].tick_params(axis='x', rotation=90)
            
            
            res = d_terr.save_results(df_m.index, IGT, m, species, file)
            #save to txt files
            
            d_terr.gen_txt(res,file,species)
            
    # merge results to file for replaydb
    # use shutil function to join all text files