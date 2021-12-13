# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 12:14:32 2021

@author: George
"""

#imports
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

os.chdir('MODS')
import dataread_terrain as d_terr
os.chdir('..')

#test main script for desired file
root = r'C:\Users\George\Documents\SETAC\Terrain'
files = ['toxmate_1.csv','toxmate_2.csv','toxmate_3.csv']

species = 'GER'
data = d_terr.main(files,species,path = root,merge = True)

#%%

#superpose plots
colours = {'G':'tab:green','E':'tab:blue','R':'tab:orange'}

plt.figure()
plt.ylim((0,100))
#plt.set_ylabel('?')
#plt.set_xlable('?')

for spec in species:
    IGT = data[spec]['IGT']
    plt.plot(IGT,color = colours[spec],alpha = 0.7)
    #plt.fill_between(np.arange(len(IGT)), IGT, color = colours[spec], alpha = 0.3)
    
#set xticks using dates from df_m (one date per month)

