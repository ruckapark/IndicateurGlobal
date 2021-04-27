# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 16:44:49 2021

Read terrain data

@author: Admin
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import dataread_terrain as d_terr
from datetime import timedelta
from data_merge import merge_dfs

plt.close('all')


### Functions

def bruit_de_fond(values,species):
    """ 
    Quantile bruit de fond
    Facteur division
    """
    seuil_bdf = {'G':[0.7,19],'E':[0.7,18],'R':[0.8,5]}
    
    # quantile / facteur
    bdf = np.nanquantile(values,seuil_bdf[species][0])/seuil_bdf[seuil_bdf][1]
    if np.isnan(bdf):
        bdf = 0
    return bdf

cutoff = {'G':[2000,3500,12000],'E':[1000,2500,10000],'R':[250,450,1200]}
offsets = {'G':3120,'E':1869,'R':406} #parametres optimises pour cannes
def IGT_percent(IGT,species):
    
    """  """
    
    seuil  = cutoff[species]
    offset = offsets[species]
    
    if IGT < seuil[0]:
        return (IGT / (seuil[0]/45))
    elif IGT < seuil[1]:
        return ((IGT - seuil[0]) / ((seuil[1] - seuil[0])/25)) + 45
    else:
        return (np.log(IGT - offset) - np.log(seuil[1] - offset)) * (20 / np.log((seuil[2] - offset)/(seuil[1]-seuil[0]))) + 70 
    


### Main

if __name__ == '__main__':

    colors = [
        '#42f5e0','#12aefc','#1612fc','#6a00a3',
        '#8ef743','#3c8f01','#0a4001','#fc03ca',
        '#d9d200','#d96c00','#942c00','#fc2803',
        '#e089b6','#a3a3a3','#7a7a7a','#303030'
        ]
    
    root = r'D:\VP\Viewpoint_data\Suez'
    os.chdir(root)
    files = [f for f in os.listdir() if '1802_1703.csv' in f]
    
    print('The following files will be studied:')
    print(files)
    
    dfs,dfs_mean = d_terr.read_data_terrain(files)
    
    #%%
    
    species = 'G'
    df = dfs[species]
    df_mean = dfs_mean[species]
    
    
    # plot all on same figure - no mean and mean
    d_terr.single_plot(df,species,title = 'Distance covered')
    d_terr.single_plot(df_mean,species,title = 'Distance covered movingmean')
    
    # plot individually (plot16)
    fig,axe = d_terr.plot_16(df_mean)    
        
    # means
    plt.figure()
    plt.plot(df.index,df.mean(axis = 1))
    
    #find the distribution of all the columns
    values = np.array(df)
    values.sort()
    
    #search data for organisms
    data_alive,data_counters = d_terr.search_dead(np.array(df),species)
    m = np.ones(len(data_alive),dtype = float) - (np.sum(data_alive,axis = 1))/16
    
    #plot mortality
    #d_terr.single_plot(m, species)
    
    #find all values in csv (array [timebins,16])
    data = np.array(df)
    data[data_alive == 0] = np.nan
    
    #calculate classic IGT and remove nan
    IGT = np.nanquantile(data,0.05,axis = 1)**2
    IGT = np.nan_to_num(IGT)
    
    """
    
    Dev. algo bruit de fond
    
    for i in range(21):
        q = 0.05 * i
        quantiledata = np.nanquantile(values,q,axis = 1)
        quantiledata.sort()
        fig = plt.figure(figsize = (15,8))
        axe = fig.add_axes([0.1,0.1,0.8,0.8])
        axe.plot(quantiledata[quantiledata > 0])
        axe.set_title('quantile {}, {}perc nonzero'.format(q,(len(quantiledata[quantiledata > 0])/len(quantiledata))))
    
    """
        
    #variation bruit de fond - 0-5%
    seuil_bdf = {
        'G':[0.7,19],
        'E':[0.7,18],
        'R':[0.8,5]
        }
    bdf = np.array(pd.Series(np.nanquantile(data,seuil_bdf[species][0],axis = 1)/seuil_bdf[species][1]).rolling(10).mean().fillna(0))
    
    #%%ajout - pourcentage avec q(0.05)
    paliers = {
        'G':[2000,3500],
        'E':[1000,2500],
        'R':[250,450]
        }
    
    
    
    palier = paliers[species]
    IGT_low,IGT_mid,IGT_high = IGT[IGT < palier[0]],IGT[(IGT >= palier[0]) & (IGT < palier[1])],IGT[IGT >= palier[1]]
    percentage = []
    
    for i in range(len(IGT)):
        if species == 'R':
            #strategy low - direct transfer from 0-250 -> 5-50%
            if IGT[i] < palier[0]:
                percentage.append(IGT[i]/5.55)
            elif IGT[i] < palier[1]:
                percentage.append((IGT[i] - palier[0])/8 + 45)
            else:
                percentage.append((np.log(IGT[i] - 406) - np.log(44))*(20 / (np.log(1200 - 406) - np.log(44))) + 70)
                
        if species == 'G':
            if IGT[i] < palier[0]:
                percentage.append(IGT[i]/ 44.44)
            elif IGT[i] < palier[1]:
                percentage.append((IGT[i] - palier[0])/60 + 45)
            else:
                percentage.append((np.log(IGT[i] - 3120) - np.log(3500 - 3120))*(20 / (np.log(12000 - 3120) - np.log(3500 - 3120))) + 70)
                
        if species == 'E':
            if IGT[i] < palier[0]:
                percentage.append(IGT[i]/ 22.22)
            elif IGT[i] < palier[1]:
                percentage.append((IGT[i] - palier[0])/60 + 45)
            else:
                percentage.append((np.log(IGT[i] - 1869) - np.log(2500 - 1869))*(20 / (np.log(10000 - 1869) - np.log(2500 - 1869))) + 70)
                
    
    percentage = np.array(bdf + np.array(percentage))
    
    fig,axe = plt.subplots(2,1,sharex = True,figsize = (20,10))
    axe[0].plot(df.index,IGT)
    axe[1].plot(df.index,percentage)
    
    