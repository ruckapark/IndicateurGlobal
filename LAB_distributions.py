# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 19:11:27 2023

Get the distribution of all mean and IGT values for 'all distibutions'

@author: George
"""

#%% IMPORTS

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%% IMPORT personal mods
os.chdir('MODS')
import dataread as d_
os.chdir('..')

import LAB_ToxClass as TOX

#%% main code
if __name__ == '__main__':
    
    specie = {'E': 'Erpobdella','G':'Gammarus','R':'Radix'}
    directories = {s:[] for s in specie}
    
    data_mean = {s:np.array([],dtype = float) for s in specie}
    data_qlow = {s:np.array([],dtype = float) for s in specie}
    data_qhigh = {s:np.array([],dtype = float) for s in specie}
    
    #collect relevant roots eventually looping through species
    for s in specie:
        for Tox in range(760,769):
            #if Tox == 766: continue #temoin
            base = r'I:\TXM{}-PC'.format(Tox)
            for d in [direc for direc in os.listdir(base) if os.path.isdir(r'{}\{}'.format(base,direc))]:
                filename = '{}_{}.csv.zip'.format(d.split('-')[0],specie[s])
                if os.path.isfile(r'{}\{}\{}'.format(base,d,filename)):
                    directories[s].append(r'{}\{}'.format(base,d))
    
    #only take where exists for all
    roots = [f for f in directories['E'] if f in directories['G']]
    roots = [r for r in roots if r in directories['R']]
    
    for root in roots:
        
        try:
            data = TOX.csvDATA(root)
        except:
            print('oh no')
            continue
        
        for s in data.active_species:
            mean = data.mean_short[s]
            IGT = data.IGT_short[s]
            IGT_minus = IGT[IGT < 0]
            IGT_plus = IGT[IGT > 0]
            
            data_mean[s] = np.hstack((data_mean[s],mean.values))
            data_qlow[s] = np.hstack((data_qlow[s],IGT_minus.values))
            data_qhigh[s] = np.hstack((data_qhigh[s],IGT_plus.values))
    
    #%% Plots
    plt.close('all')
    
    for s in specie:
        
        TOX.ToxPLOT(data_mean[s]).plotHIST(title = 'Mean data {}'.format(specie[s]))
        TOX.ToxPLOT(data_qlow[s]).plotHIST(title = 'Qlow data {}'.format(specie[s]))
        TOX.ToxPLOT(data_qhigh[s]).plotHIST(title = 'qhigh data {}'.format(specie[s]))