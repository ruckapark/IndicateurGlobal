# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 16:22:50 2023

@author: George
"""

import os
import pandas as pd
import numpy as np

#%% IMPORT personal mods
os.chdir('MODS')
from dope_reg import dope_read_extend
os.chdir('..')

dope_df = dope_read_extend()

molecules = dope_df['Substance'].unique()
df_mols = pd.DataFrame(index = np.arange(molecules.shape[0]),columns = ['Name','Formula','NoTests','Concentrations','Class','SubClass'])
df_mols['Name'] = molecules

for i in range(df_mols.shape[0]):
    
    #update formula
    mol = df_mols.iloc[i]['Name']
    tests = dope_df[dope_df['Substance'] == mol]
    df_mols['Formula'].iloc[i] = tests.iloc[0]['Molecule']
    df_mols['NoTests'].iloc[i] = tests.shape[0]
    df_mols['Concentrations'].iloc[i] = tests['Concentration'].unique()
    
df_mols.to_csv(r'D:\VP\Viewpoint_data\REGS\molecules.csv',index = False)