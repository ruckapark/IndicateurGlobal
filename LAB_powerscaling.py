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
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_time_series_matplotlib(time_series_list):
    """
    Plot N time series on a 3D axis using Matplotlib with customization.

    Parameters:
    - time_series_list: List of pandas Series or numpy arrays representing time series data.

    Returns:
    - 3D plot using Matplotlib with specified customization.
    """
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    for i, series in enumerate(time_series_list):
        offset = np.ones_like(series) * i
        ax.plot(offset, range(len(series)), series, color='b', label=f'Time Series {i + 1}')

        # Fill between the curve and z=0
        ax.plot_surface(np.array([offset, offset]),
                        np.array([range(len(series)), range(len(series))]),
                        np.array([series, np.zeros_like(series)]),
                        color='b', alpha=0.5)
        
    tmp_planes = ax.zaxis._PLANES 
    ax.zaxis._PLANES = ( tmp_planes[2], tmp_planes[3], 
                         tmp_planes[0], tmp_planes[1], 
                         tmp_planes[4], tmp_planes[5])
    ax.set_xlabel('Offset')
    ax.set_ylabel('Time')
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel('Measured Quantity', rotation=90)
    
    #plot line at z=0
    ax.plot([0, 0], [-1, 11], [0, 0],color = 'black')   # extend in y direction
    
    ax.set_ylim(0,10)

    ax.grid(color='white', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    ax.grid(False)

    plt.show()

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