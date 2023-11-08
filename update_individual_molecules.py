# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 11:54:41 2023

Create individual file with all the root/file locations for each repetition of the molecules

copper.csv

repetition date - concentration - root/file gammarus - root/file temoin g - root/file erpobdella - root/file temoin e - root/file radix - root/file temoin r
repetition date - concent...

The idea is then to create a custom folder with NaNs where there is no data available

@author: George
"""

import os
import pandas as pd
import numpy as np

#%% IMPORT personal mods
os.chdir('MODS')
from dope_reg import dope_read_extend
os.chdir('..')

def find_closest_timestamp(timestamp,series):
    return series.iloc[np.argmin([np.abs((t - timestamp).total_seconds()) for t in series])]

def find_filename(Tox,root,species):
    
    part1 = Tox
    part2 = root
    part3 = root.split('-')[0] + '_' + species
    return r'I:\TXM{}-PC\{}\{}.csv.zip'.format(part1,part2,part3)

specie = {'E':'Erpobdella','G':'Gammarus','R':'Radix'}

dope_df = dope_read_extend()
molecules = dope_df['Substance'].unique()

#define temoins
temoins = dope_df[dope_df['Substance'] == 'TEM']
temoins = temoins[(temoins['End'] < pd.to_datetime('2021-03-12',format = '%Y-%m-%d')) | (temoins['End'] > pd.to_datetime('2021-03-16',format = '%Y-%m-%d'))]

#remove TEMOIN
molecules = list(molecules)
molecules.remove('TEM')

#df columns for individual files
cols = ['Concentration','Erpobdella','Erpobdella_TEM','Gammarus','Gammarus_TEM','Radix','Radix_TEM']

root = r'D:\VP\Viewpoint_data\REGS\Molecules'

for m in molecules:
    
    #all repetitions
    tests = dope_df[dope_df['Substance'] == m]
    
    #Check if directory already exists and locate
    if m == 'Cuivre': m = 'Copper'
    
    #create df with molecule
    df = pd.DataFrame(index = np.arange(tests.shape[0]),columns = cols)
    df.index.name = 'Repetition'
    
    for i in range(df.shape[0]):
        
        #find temoin closest to this entry
        tem_timestamp = find_closest_timestamp(tests.iloc[i]['End'],temoins['End'])
        tem = temoins[temoins['End'] == tem_timestamp].iloc[0]
        
        df.loc[i]['Concentration'] = tests.iloc[i]['Concentration']
        
        # #Commented are names for individual files
        # df.loc[i]['Erpobdella'] = find_filename(tests.iloc[i]['TxM'],tests.iloc[i]['root'][-1],'Erpobdella')
        # df.loc[i]['Erpobdella_TEM'] = find_filename(tem['TxM'],tem['root'][-1],'Erpobdella')
        # df.loc[i]['Gammarus'] = find_filename(tests.iloc[i]['TxM'],tests.iloc[i]['root'][-1],'Gammarus')
        # df.loc[i]['Gammarus_TEM'] = find_filename(tem['TxM'],tem['root'][-1],'Gammarus')
        # df.loc[i]['Radix'] = find_filename(tests.iloc[i]['TxM'],tests.iloc[i]['root'][-1],'Radix')
        # df.loc[i]['Radix_TEM'] = find_filename(tem['TxM'],tem['root'][-1],'Radix')
        
        for s in specie:
            df.loc[i][specie[s]] = r'I:\TXM{}-PC\{}'.format(tests.iloc[i]['TxM'],tests.iloc[i]['root'][-1])
            df.loc[i]['{}_TEM'.format(specie[s])] = r'I:\TXM{}-PC\{}'.format(tem['TxM'],tem['root'][-1])
    
    df.to_csv('{}\{}.csv'.format(root,m))