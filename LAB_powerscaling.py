# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 09:25:53 2023

Code to illustrate semi qualitative power scaling of data before normalisation for high and low quantiles

Illustrate results using copper repetitions for all species

@author: George
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import LAB_ToxClass as TOX

def powerlimit_scale(X, y=0.85, p=0.5):
    """
    Perform power scaling for values above the limit 'y'.

    Parameters:
    - X: numpy array or pandas series
    - y: threshold value
    - p: power scaling factor

    Returns:
    - Power-scaled array or series
    """
    # Convert to numpy array if input is pandas series
    series = False
    if isinstance(X, pd.Series):
        series = True
        index = pd.Index(X.index, name=X.index.name)
        X = X.values

    # Apply power scaling only to values above the threshold 'y'
    X_scaled = np.where(X > y, y + (X + 1 - y)**p - 1, X)

    # If the input was a pandas series, return the result as a series
    if series:
        X_scaled = pd.Series(X_scaled, index=index)
    
    return X_scaled

if __name__ == '__main__':
    
    values = np.linspace(0,2.5,60)
    plt.figure()
    plt.plot(values)
    plt.plot(powerlimit_scale(values),color = 'orange')
    
    specie = {'E':'Erpobdella','G':'Gammarus','R':'Radix'}
    root_dir = r'D:\VP\Viewpoint_data\REGS\Molecules'
    m = 'Copper'
    
    #read selected directories
    df = pd.read_csv(r'D:\VP\Viewpoint_data\REGS\Molecules\{}_custom.csv'.format(m),index_col = 'Repetition')
    
    #for 3 species plot each IGT curve and label the associated repetition
    fig,axe = plt.subplots(1,3,figsize = (18,8),sharex = True)
    for i,s in enumerate(specie): 
        axe[i].set_title(specie[s])
        axe[i].set_ylim((-2,2))
    
    for i in range(df.shape[0]):
        entry = {s:df[specie[s]] for s in specie}
        data = TOX.csvDATA_comp({s:df[specie[s]].iloc[i] for s in specie})
        for x,s in enumerate(data.species):
            axe[x].plot(data.IGT[s],label = i)
    
    axe[x].legend()
    
    #for 3 species plot each IGT curve and label the associated repetition
    fig,axe = plt.subplots(1,3,figsize = (18,8),sharex = True)
    for i,s in enumerate(specie): 
        axe[i].set_title(specie[s])
        axe[i].set_ylim((-2,2))
    
    for i in range(df.shape[0]):
        entry = {s:df[specie[s]] for s in specie}
        data = TOX.csvDATA_comp({s:df[specie[s]].iloc[i] for s in specie})
        for x,s in enumerate(data.species):
            axe[x].plot(powerlimit_scale(data.IGT[s]),label = i)
    
    axe[x].legend()