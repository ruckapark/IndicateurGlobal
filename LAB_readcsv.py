# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 14:55:28 2023

csv files for individual species - already treated read from the NAS

@author: George
"""

#%% IMPORTS

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

def preproc_csv(df):
    df.index = pd.to_datetime(df.index)
    return df

#%% main code
if __name__ == '__main__':
    
    specie = {'E': 'Erpobdella','G':'Gammarus','R':'Radix'}
    dfs = {}
    root = r'I:\TXM762-PC\20220225-090938'
    rootfile_stem = root + r'\\' + root.split('\\')[-1].split('-')[0] + '_'
    
    try:
        morts = d_.read_dead(root)
    except:
        print('No mortality register')
        os._exit()
        
    for s in specie:
        species = specie[s]
        if len(morts[s]) > 11:
            print('\n Excessive mortality: ',specie)
            continue
        
        df = pd.read_csv(r'{}{}.csv.zip'.format(rootfile_stem,species),index_col = 0)
        dfs.update({s:preproc_csv(df)})
        
    #Check for mortality in data as double check
    