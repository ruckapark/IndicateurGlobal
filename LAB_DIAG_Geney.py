# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 17:19:55 2024

Bioessaie example

@author: George
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import LAB_ToxClass_bioessai_test as TOX
from MODS.dope_reg import dope_read_extend

sns.set_style('white')

bioessais = {
    'Test1':[r'I:\TXM760-PC\20231215-163118',r'I:\TXM760-PC\20231216-082344'],
    'Test2':[r'I:\TXM761-PC\20231215-163139',r'I:\TXM761-PC\20231216-082334'],
    'Test3':[r'I:\TXM762-PC\20231215-163339',r'I:\TXM762-PC\20231216-082510']}

def return_grids():
    
    fig,axes = plt.subplots(3,1,figsize = (8,8),sharex = True)
    return fig,axes

def get_ylimits(axe):
    
    #fit within -0.8 and 0.8 unless already higher
    lims = axe.get_ylim()
    axe_lims = [-0.6,1.5]
    if lims[0] < axe_lims[0]:
        axe_lims[0] = lims[0]
    if lims[1] > axe_lims[1]:
        axe_lims[1] = lims[1] 
        
    return axe_lims

if __name__ == "__main__":
    
    #read in bioessai as dope register and retrieve data
    dope_df = dope_read_extend('bioessai_reg')
    data = TOX.csvDATA(bioessais['Test3'][1],dope_df)
    
    IGT = data.IGT_
    mean = data.mean_
    
    #plot figures - for annex of report
    fig,axes = plt.subplots(3,1,figsize = (8,8),sharex = True)
    for i,s in enumerate(data.species):
        ind = IGT[s].index / 60 
        axes[i].plot(ind,IGT[s].values,color = data.species_colors[s])
        axes[i].set_title(data.species[s])
        axes[i].set_ylim(get_ylimits(axes[i]))
        
    #Scores based on which species?
    
    
    