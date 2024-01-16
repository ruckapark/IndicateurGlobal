# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 15:06:03 2024

@author: George
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from sklearn import preprocessing
from datetime import timedelta

#%% IMPORT personal mods
import LAB_ToxClass as TOX
os.chdir('MODS')
from data_merge import merge_dfs
from dope_reg import dope_read_extend
import dataread as d_
os.chdir('..')

#%% Set plotting style to whit grid
sns.set_style("white")

#%% Exceptions
# bad_temoins = {
#     'Copper': {'E':[5,6,7],'G':[2],'R':[0,1,6,7]},
#     'Methomyl': {'E':[1,2],'G':[],'R':[]},
#     'Verapamil': {'E':[2],'G':[],'R':[]},
#     'Zinc': {'E':[2,3],'G':[],'R':[]}}

#IGT
# bad_temoins = {
#     'Copper': {'E':[5,6,7],'G':[2],'R':[0,1,6,7]},
#     'Methomyl': {'E':[1,2],'G':[],'R':[]},
#     'Verapamil': {'E':[2],'G':[],'R':[]},
#     'Zinc': {'E':[2,3],'G':[],'R':[]}}

#%%
if __name__ == '__main__':
    plt.close('all')

    #Read data and plot
    specie = {'E':'Erpobdella','G':'Gammarus','R':'Radix'}
    substance = 'Zinc'
    root = r'D:\VP\ARTICLE2\ArticleRawData\{}'.format(substance)
    colors = {'E': '#de8e0d', 'G': '#057d25', 'R': '#1607e8'}
    figures = {s:plt.figure(figsize = (10,6)) for s in specie}
    axes = {s:figures[s].add_axes([0.1,0.1,0.8,0.8]) for s in specie}
    
    extension = 'IGT'
    #extension = 'mean' #divide by three
    
    if extension == 'mean':
        bad_temoins = {
            'Copper': {'E':[5,6,7],'G':[2],'R':[0,1,6,7]},
            'Methomyl': {'E':[1,2],'G':[],'R':[]},
            'Verapamil': {'E':[2],'G':[],'R':[]},
            'Zinc': {'E':[2,3],'G':[],'R':[]}}
    else:
        bad_temoins = {
            'Copper': {'E':[5,6,7],'G':[2],'R':[0,1,6,7]},
            'Methomyl': {'E':[1,2,3],'G':[1,2,5,6],'R':[]},
            'Verapamil': {'E':[2],'G':[],'R':[]},
            'Zinc': {'E':[3],'G':[0],'R':[]}}
    
    for s in specie:
        no_data = len(os.listdir(r'{}\{}'.format(root,specie[s])))//4
        for i in range(no_data):
        #for i in [0,1,2,3]:
            if extension == 'mean':
                if s == 'R':
                    factor = 0.4
                elif (s == 'G') and (substance == 'Zinc'):
                    factor = 0.5
                else:
                    factor = 1
            else:
                factor = 1
            
            data = d_.read_csv_to_series(r'{}\{}\{}{}_{}.csv'.format(root,specie[s],substance,i,extension))
            data_ = d_.read_csv_to_series(r'{}\{}\{}{}_{}_.csv'.format(root,specie[s],substance,i,extension))
            
            #moving mean
            window_size = 5*3 #5minutes
            data = data.rolling(window=window_size, center=False).mean()
            data_ = data_.rolling(window=window_size, center=False).mean()
            
            axes[s].plot(data.index/60,data.values,color = colors[s])
            
            if i in bad_temoins[substance][s]: continue
        
            axes[s].plot(data_.index/60,data_.values*factor,color = '#737373')
            
            
            #plot parameters
            axes[s].set_title(specie[s],fontsize = 18)
            axes[s].set_xlabel('Time post exposure (minutes)',fontsize = 16)
            axes[s].set_ylabel('Normalised {}'.format(extension),fontsize = 16)
            axes[s].axvline(0,color = 'red',linestyle = '--')
            
            legend_handles = [
                Line2D([0], [0], color=colors[s], label='Spike curves'),
                Line2D([0], [0], color='black', label='Control curves'),
                Line2D([0], [0], color='red', label='Spike', linestyle = '--')
                ]
            
            axes[s].legend(handles=legend_handles,fontsize = 16)