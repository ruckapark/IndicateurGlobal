# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 15:21:28 2022

File to describe in detail the data of each species for a terrain data file

Depends on : TERRAIN_readdata.py & dataread_terrain.py

@author: GD0llo
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%% IMPORT personal mods
os.chdir('MODS')
import dataread_terrain as d_terr
os.chdir('..')
import TERRAIN_readdata as d_

if __name__ == '__main__':
    
    # Read in data file qnd run readdata to obtain data files and form basic plots
    path = r'D:\VP\Viewpoint_data\TERRAIN\Eawag'
    file = ['toxmate_17012022.xls']
    
    #thresholds not working
    data,thresholds = d_.main(file,'GR',root = path,thresholds = None)
    
    #%% Inspect data
    
    #check IGT calculation with raw values for 10 values
    # ind_plot = 
    sp = 'R'
    df = data[sp]['df']
    means = df.mean(axis = 1)
    plt.figure()
    plt.plot(means)
    
    #individual plots
    d_terr.plot_16(data['G']['df'])
    
    #distplot
    fig = plt.figure(figsize = (14,8))
    axe = fig.add_axes([0.1,0.1,0.8,0.8])
    sns.histplot(np.array(df),bins = 500,ax = axe)
    axe.set_xlim((0,np.nanquantile(np.array(df),0.98)))
    axe.axvline(np.nanquantile(np.array(df),0.97),color = 'r',linestyle = '--')
    axe.set_title('Distribution - {}'.format(sp))
    
    print(np.nanquantile(np.array(df),0.97))
    
    #Check mortality values correspond
    
    #Check what happens
    