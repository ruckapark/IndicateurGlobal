# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 11:18:05 2021

@author: George
"""

#imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 300})

#read data file
means = pd.read_csv(r'C:\Users\George\Documents\SETAC\DATA_means.csv')
IGT = pd.read_csv(r'C:\Users\George\Documents\SETAC\DATA_IGT.csv')
IGT.index = (np.array(IGT.index) - 178)/3

#substances
substances = {sub.split('_')[1] for sub in means.columns}

#colours
colours = ['tab:blue','tab:orange','tab:green','tab:red','tab:pink','tab:olive','tab:cyan']
fig = plt.figure()
ax = fig.add_axes([0.15,0.15,0.75,0.75])

#extract individual substance for development
for i,sub in enumerate(substances):
    IGT_t = IGT[[col for col in IGT.columns if sub in col]]
    IGT_t['MEAN'] = IGT_t.mean(axis = 1)
    
    if sub == '124-Trichlorobenzene':
        IGT_t['MEAN'] = IGT_t['MEAN']*2
    elif sub == 'diclofenac':
        IGT_t['MEAN'] = IGT_t['MEAN']*3
    
    #plot graph
    ax.plot(IGT_t['MEAN'],colours[i],label = sub)
    
    """
    for col in IGT_t.drop(columns = 'MEAN').columns:
        ax.plot(IGT_t[col],color = colours[i],linestyle = '--',alpha = 0.5)
    """

ax.set_xlabel('Time post-contamination (minutes)')
ax.set_ylabel('Stress related activity in displacement \newline per unit time (mm{}/s)'.format('$^2$'))
ax.set_title('Bio-activity response of Gammarus Fossarum to various contaminants')
ax.axvline(0,color = 'black',linestyle = '--')        
ax.legend()