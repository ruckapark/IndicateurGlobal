# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 14:44:18 2023

Remove single peaks in the data if possible

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
import LAB_splines as SPL

os.chdir('MODS')
from data_merge import merge_dfs
from dope_reg import dope_read_extend
import dataread as d_
os.chdir('..')

testsets = df = pd.read_csv(r'D:\VP\Viewpoint_data\REGS\Molecules\Zinc.csv',index_col = 'Repetition')

def find_switching_indices(series):
    # Convert the pandas series to a numpy array
    array_values = np.array(series)    
    # Find the indices where the signal switches from 0 to 1
    switching_indices_positive = np.where(np.diff(array_values) == 1)[0] + 1
    switching_indices_negative = np.where(np.diff(array_values) == -1)[0] + 1
    return switching_indices_positive,switching_indices_negative

#%%
if __name__ == "__main__":
    
    data = TOX.csvDATA(r'I:\TXM760-PC\20210513-230436')
    
    #%% data treatment
    plt.close('all')
    
    #by time stamp if most around a central point are zero, then filter to zero
    for s in data.species:
        plt.figure()
        plt.plot(data.q_high_adj_short[s])
        
    species = 'R'
    
    df = data.q_high_adj_short[species]
    
    plt.figure()
    plt.plot(data.q_high_adj_short[species])
    
    #find continuous areas of zero
    bin_arr = df.copy()
    bin_arr = 1*(bin_arr < -0.1)
    
    #assume there are multiple values in the array
    switch_on,switch_off = find_switching_indices(bin_arr)
    if switch_off[0] < switch_on[0]: switch_off = switch_off[1:]
    
    print(switch_off - switch_on)