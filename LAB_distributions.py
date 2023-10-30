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
    
    fig,axe = plt.subplots(1,3,figsize = (20,7))
    fig.suptitle('Mean Activity Distribution',fontsize = 18)
    fig.text(0.5, 0.04, 'Distance', ha='center',fontsize = 16)
    for i,s in enumerate(specie):
        
        histdata = np.array(data_mean[s])
        histdata = histdata.flatten()
        
        sns.histplot(histdata,ax=axe[i],color = data.species_colors[s],kde = True)
        axe[i].set_title(data.species[s],fontsize = 16)
        if s == 'E': axe[i].set_xlim((-5,150))
        
        q75 = np.abs(np.quantile(histdata,0.75))
        axe[i].axvline(q75,color = 'black',linestyle = '--',linewidth = 1.5,label = 'Upper quartile')
        
        if i == 2: axe[i].legend(fontsize = 17)
        
    fig,axe = plt.subplots(1,3,figsize = (20,7))
    fig.suptitle('High Quantile Activity Distribution',fontsize = 18)
    fig.text(0.5, 0.04, 'Distance', ha='center',fontsize = 16)
    for i,s in enumerate(specie):
        
        histdata = np.array(data_qlow[s])
        histdata = histdata.flatten()
        
        sns.histplot(histdata,ax=axe[i],color = data.species_colors[s])
        axe[i].set_title(data.species[s],fontsize = 16)
        
        q90 = np.quantile(histdata,0.1)
        axe[i].axvline(q90,color = 'black',linestyle = '--',linewidth = 1.5,label = 'Quantile -0.9')
        
        if i == 1: axe[i].legend(fontsize = 17)
        
    fig,axe = plt.subplots(1,3,figsize = (20,7))
    fig.suptitle('Low Quantile Activity Distribution',fontsize = 18)
    fig.text(0.5, 0.04, 'Distance', ha='center',fontsize = 16)
    for i,s in enumerate(specie):
        
        histdata = np.array(data_qhigh[s])
        histdata = histdata.flatten()
        
        sns.histplot(histdata,ax=axe[i],color = data.species_colors[s])
        axe[i].set_title(data.species[s],fontsize = 16)
        
        axe[i].set_ylim((0,3000))
        
        q95 = np.quantile(histdata,0.95)
        axe[i].axvline(q95,color = 'black',linestyle = '--',linewidth = 1.5,label = 'Quantile 0.95')
        
        if i == 2: axe[i].legend(fontsize = 17)
        
    fig = plt.figure(figsize = (13,7))
    axe = fig.add_axes([0.1,0.1,0.8,0.8])
    axe.set_title('Scaled Low Quantile Activity Distributions',fontsize = 18)

    for i,s in enumerate(specie):
        
        histdata = data_qlow[s].copy()
        histdata = histdata.flatten()/np.abs(np.quantile(data_qlow[s],0.1))
        sns.histplot(histdata,ax=axe,color = data.species_colors[s],alpha = 1-0.3*i,label = data.species[s])
        
    axe.legend(fontsize = 18)
    fig.text(0.5, 0.04, 'Scaled Distance', ha='center',fontsize = 16)


    fig = plt.figure(figsize = (13,7))
    axe = fig.add_axes([0.1,0.1,0.8,0.8])
    axe.set_title('Scaled High Quantile Activity Distributions',fontsize = 18)

    for i,s in enumerate(specie):
        
        histdata = data_qhigh[s].copy()
        histdata = histdata.flatten()/np.abs(np.quantile(data_qhigh[s],0.95))
        sns.histplot(histdata,ax=axe,color = data.species_colors[s],alpha = 1-0.3*i,label = data.species[s])

    axe.legend(fontsize = 18)
    axe.set_xlim(0,1.5)
    axe.set_ylim(0,8000)
    fig.text(0.5, 0.04, 'Scaled Distance', ha='center',fontsize = 16)
        
    fig = plt.figure(figsize = (13,7))
    axe = fig.add_axes([0.1,0.1,0.8,0.8])
    axe.set_title('Scaled Mean Activity Distributions',fontsize = 18)

    for i,s in enumerate(specie):
        
        histdata = data_mean[s].copy()
        histdata = histdata.flatten()/np.abs(np.mean(data_mean[s]))
        sns.histplot(histdata,ax=axe,color = data.species_colors[s],alpha = 1-0.3*i,label = data.species[s])
     
    axe.legend(fontsize = 18)
    axe.set_xlim((-0.25,4))
    fig.text(0.5, 0.04, 'Scaled Distance', ha='center',fontsize = 16)