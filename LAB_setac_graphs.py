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

plt.rcParams.update({'font.size': 10})

#read data file
means = pd.read_csv(r'C:\Users\George\Documents\SETAC\DATA_means.csv')
IGT = pd.read_csv(r'C:\Users\George\Documents\SETAC\DATA_IGT.csv')
IGT.index = (np.array(IGT.index) - 178)/3

#substances
substances = {sub.split('_')[1] for sub in means.columns}

#colours
colours = ['tab:blue','tab:orange','tab:green','tab:red','tab:pink','tab:olive','tab:cyan']
concentrations = ['1000ug','125ug','75ug','100ug','20ug','1500ug','100ug']
fig = plt.figure(figsize = (25,8))
ax = fig.add_axes([0.15,0.15,0.75,0.75])

IGT = IGT[IGT.index > -25]

#extract individual substance for development
for i,sub in enumerate(substances):
    IGT_t = IGT[[col for col in IGT.columns if sub in col]]
    IGT_t['MEAN'] = IGT_t.mean(axis = 1)
    
    if sub == '124-Trichlorobenzene':
        IGT_t['MEAN'] = IGT_t['MEAN']*2
    elif sub == 'diclofenac':
        IGT_t['MEAN'] = IGT_t['MEAN']*3
    
    #plot graph
    ax.plot(IGT_t['MEAN'],colours[i],label = '{} - {}'.format(concentrations[i],sub))
    
    """
    for col in IGT_t.drop(columns = 'MEAN').columns:
        ax.plot(IGT_t[col],color = colours[i],linestyle = '--',alpha = 0.5)
    """

ax.set_xlabel('Time post-contamination (minutes)',fontsize=17)
ax.set_ylabel('Stress related activity in displacement \n per unit time (mm{}/s)'.format('$^2$'),fontsize=17)
ax.set_title('Bio-activity response of Gammarus Fossarum to various contaminants',fontsize=20)
ax.axvline(0,color = 'black',linestyle = '--')        
ax.legend(fontsize=17)