# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 11:51:51 2023

Compare results of datafiles from the replay versions of files

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
os.chdir('MODS')
from data_merge import merge_dfs
from dope_reg import dope_read_extend
import dataread as d_
os.chdir('..')

#%% Relevant directories
roots = ['765_20211022',
         '767_20211022',
         '762_20211022',
         '763-20211022',
         '762_20211028',
         '762_20220225',
         '763_20220225',
         '764_20220310',
         '765_20220310',
         '765_20220317']


#%% Code

if __name__ == '__main__':
    
    specie = {'E': 'Erpobdella','G':'Gammarus','R':'Radix'}
    
    root = roots[0]
    Tox = root.split('_')[0]
    
    stem = [r for r in os.listdir(r'I:\TXM{}-PC'.format(Tox)) if root.split('_')[-1] in r]
    root = r'I:\TXM{}-PC\{}'.format(Tox,stem[0])
    
    #locate original and copy
    file_og = r'{}\{}.xls.zip'.format(root,stem[0])
    file_copy = r'{}\{}.replay.xls.zip'.format(root,stem[0])
    
    #read file
    df_og,df_copy = d_.read_merge([file_og]),d_.read_merge([file_copy])
    dfs_og,dfs_copy = d_.preproc(df_og),d_.preproc(df_copy)
    
    #register
    #dope_df = dope_read_extend()
    
    #%%
    for i in range(1,17):
        plt.figure()
        plt.plot(np.arange(dfs_og['R'].shape[0]),dfs_og['R'][i])
        plt.plot(np.arange(dfs_copy['R'].shape[0]),dfs_copy['R'][i])