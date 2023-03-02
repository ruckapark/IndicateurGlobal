# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 11:02:19 2023

create database for ToxPrints log - start by reading in 

@author: George
"""

import sys
import numpy as np
import pandas as pd

sys.path.insert(0, r'D:\VP\Viewpoint_data\code')
import MODS.dope_reg as dope_reg

if __name__ == '__main__':
    
    #read in original dope reg
    reg = dope_reg.dope_read()
    
    #find corresponding file roots
    allfiles = pd.read_csv('allfiles.txt',delimiter = ',',names = ['root','Tox'])
    allfiles['datetime'] = pd.to_datetime(allfiles['root'],format = '%Y%m%d-%H%M%S')
    
    #attribute dates to experiments with short / long, make list on unattributed dates
    
    reg['shortfile'] = np.nan
    reg['root'] = np.nan
    for i in range(reg.shape[0]):
        entry = reg.iloc[i]
        Tox = entry['TxM']
        dope = entry['End']
        if dope.weekday() >=4:
            dope_limit = dope - pd.Timedelta(days = dope.weekday())
        else:
            dope_limit = dope - pd.Timedelta(days = 6)
        
        files = allfiles[allfiles['Tox'] == Tox]
        files = files[(files['datetime'] < (dope + pd.Timedelta(hours = 24))) & (files['datetime'] > dope_limit)]
        
        if files.shape[0] == 1:
            reg['shortfile'].iloc[i] = 1
            reg['root'].iloc[i] = [files.iloc[0]['root']]
        else:
            reg['shortfile'].iloc[i] = 0
            reg['root'].iloc[i] = files['root'].values
            
            
    #check why some have many folders and others have none individually
    temp = reg[reg['root'].str.len() != 1]
    temp = temp[temp['root'].str.len() != 2]
    
    reg.to_csv('extended_reg.csv',index = False)