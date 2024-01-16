# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 15:06:03 2024

@author: George
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

#%%
if __name__ == '__main__':
    plt.close('all')

    #Read data and plot
    specie = {'E':'Erpobdella','G':'Gammarus','R':'Radix'}
    substance = 'Copper'
    root = r'D:\VP\ARTICLE2\ArticleRawData\{}'.format(substance)
    colors = {'E': '#de8e0d', 'G': '#057d25', 'R': '#1607e8'}
    figures = {s:plt.figure(figsize = (10,6)) for s in specie}
    axes = {s:figures[s].add_axes([0.1,0.1,0.8,0.8]) for s in specie}
    for s in specie:
        no_data = len(os.listdir(r'{}\{}'.format(root,specie[s])))//2
        for i in range(no_data):
            data = d_.read_csv_to_series(r'{}\{}\{}{}.csv'.format(root,specie[s],substance,i))
            data_ = d_.read_csv_to_series(r'{}\{}\{}{}_.csv'.format(root,specie[s],substance,i))
            
            #moving mean
            window_size = 5*3 #5minutes
            data = data.rolling(window=window_size, center=True).mean()
            data_ = data_.rolling(window=window_size, center=True).mean()
            
            axes[s].plot(data.index/60,data.values,color = colors[s])
            axes[s].plot(data_.index/60,data_.values,color = '#737373')