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
df_mols = pd.read_csv(r'D:\VP\Viewpoint_data\REGS\molecules.csv',index_col = None)

molecules = dope_df['Substance'].unique()
new_molecules = [m for m in molecules if m not in df_mols['Name'].values]

index = df_mols.shape[0]
for mol in new_molecules:
    
    df_mols.loc[index] = {}
    tests = dope_df[dope_df['Substance'] == mol]
    df_mols['Name'].iloc[index] = mol
    df_mols['Formula'].iloc[index] = tests.iloc[0]['Molecule']
    df_mols['NoTests'].iloc[index] = tests.shape[0]
    df_mols['Concentrations'].iloc[index] = tests['Concentration'].unique()
    
    index += 1
    
df_mols.to_csv(r'D:\VP\Viewpoint_data\REGS\molecules.csv',index = False)