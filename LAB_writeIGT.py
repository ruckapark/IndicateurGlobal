# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 11:28:28 2023

Read datafile from zip file and write IGT file accordingly

- Test this works for single file
- Then test for joined file

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
from dope_reg import dope_read
import dataread as d_
os.chdir('..')

#%% FUNCTIONS - plotting functions could also be put into another module         

if __name__ == '__main__':
    
    #Could be in two directories for long entry
    root = [r'I:\TXM760-PC\20210716-082422']
    Tox = int(root[0].split('TXM')[-1][:3])
    species = 'G'
    specie = {'E': 'Erpobdella','G':'Gammarus','R':'Radix'}
    
    files = []
    for r in root:
        os.chdir(root) #not necessary
        file = [file for file in os.listdir() if 'xls.zip' in file]
        files.extend(file)
     
    df = d_.read_merge(files)
    dfs = d_.preproc(df)