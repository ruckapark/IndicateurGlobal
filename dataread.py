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

#from perftimer import Timer
#t = Timer()


os.chdir(r'D:\VP\Viewpoint_data\test')

#start with only df group
df = pd.read_csv('av_dope_2500_764.csv',sep = '\t',encoding = 'utf-16')
df = df[df['datatype'] == 'Locomotion']

#sort values sn = , pn = ,location = E01-16 etcc., aname = A01-04,B01-04 etc.
df = df.sort_values(by = ['sn','pn','location','aname'])

#treat time variable
df['time'] = pd.to_datetime(df['stdate'] + " " + df['sttime'])
df = df.set_index('time')

#E01 etc.
mapping = lambda a : {'E': 'Erpobdella','G':'Gammarus','R':'Radix'}[a[0]]
df['specie'] = df['location'].map(mapping)

#moi le column 'animal' n'a que des NaNs
good_cols = ['location','stdate','specie','entct','inact','inadur','inadist','smlct','smldur','smldist','larct','lardur','lardist','emptyct','emptydur']
df = df[good_cols]

#create animal column from location E01
df['animal'] = df['location'].str[1:].astype(int)

df['dose'] = '72ug/L'
df['etude'] = 'ETUDE001'
df['lot'] = 'Zn'

isnull = df.isnull().sum()
sns.heatmap(df.isnull(), yticklabels = False, cbar = False)

#sml distance all nans? replace with zeros
df['smldist'] = df['smldist'].fillna(0)
#%%

#%%

## pre data treatment
#1. split into locomotion and quantization
# print('Groupby method:')
# t.start()
# df_loc,df_quant = [g for _, g in df.groupby('datatype')]
# t.stop()

#should check performance over the full dataset - the other method is more readable
"""
print('Explicit method:')
t.start()
df_loc = df[df['datatype']=='Locomotion']
df_quant = df[df['datatype']=='Quantization']
t.stop()
"""

#reorder? - all values present but not all in the correct order

#3. split into animal type (asign 0 - radix, 1 - gammares,2 - sangsues)


#4.

#%%
os.chdir('..')