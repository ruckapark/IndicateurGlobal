# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 10:41:44 2020

Test file for the TxM 765

@author: Admin
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from perftimer import Timer
t = Timer()


os.chdir(r'D:\VP\Viewpoint_data\test')
print(os.listdir())
files = [f for f in os.listdir() if f.endswith('.csv')]

#start with only df group
df = pd.read_csv(files[1],sep = '\t',encoding = 'utf-16')

#treat time variable
df['time'] = pd.to_datetime(df['stdate'] + " " + df['sttime'])
df = df.set_index('time')

## pre data treatment
#1. split into locomotion and quantization
print('Groupby method:')
t.start()
df_loc,df_quant = [g for _, g in df.groupby('datatype')]
t.stop()

#should check performance over the full dataset - the other method is more readable
"""
print('Explicit method:')
t.start()
df_loc = df[df['datatype']=='Locomotion']
df_quant = df[df['datatype']=='Quantization']
t.stop()
"""

#reorder? - all values present but not all in the correct order

#%%2. visualise nans for one
isnull = df_loc.isnull().sum()
#sns.heatmap(dfs_data[0].isnull(), yticklabels = False, cbar = False)

#3. split into animal type (asign 0 - radix, 1 - gammares,2 - sangsues)


#4.

#%%
os.chdir('..')