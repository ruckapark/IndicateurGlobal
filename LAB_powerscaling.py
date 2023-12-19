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
import re
#import LAB_ToxClass as TOX
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

def str_remove_num(input_string):
    """
    Remove numbers from the end of a string.

    Parameters:
    - input_string: The input string.

    Returns:
    - String with numbers removed from the end.
    """
    # Use regular expression to remove numbers from the end
    result = re.sub(r'\d+$', '', input_string)
    
    return result

def plot_3d_time_series_matplotlib(time_series_list,colors = None):
    """
    Plot N time series on a 3D axis using Matplotlib with customization.

    Parameters:
    - time_series_list: List of pandas Series or numpy arrays representing time series data.

    Returns:
    - 3D plot using Matplotlib with specified customization.
    """
    fig = plt.figure(figsize=(18,11))
    ax = fig.add_subplot(111, projection='3d')

    for i, series in enumerate(time_series_list):
        
        if colors:
            color = colors[i]
        else:
            color = 'b'
        
        offset = np.ones_like(series) * i
        ax.plot(offset, range(len(series)), series, color='black', label=f'Time Series {i + 1}')

        # Fill between the curve and z=0
        ax.plot_surface(np.array([offset, offset]),
                        np.array([range(len(series)), range(len(series))]),
                        np.array([series, np.zeros_like(series)]),
                        color=color, alpha=0.5)
        
    tmp_planes = ax.zaxis._PLANES 
    ax.zaxis._PLANES = ( tmp_planes[2], tmp_planes[3], 
                         tmp_planes[0], tmp_planes[1], 
                         tmp_planes[4], tmp_planes[5])
    ax.set_xlabel('Offset')
    ax.set_ylabel('Time')
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel('Measured Quantity', rotation=90)
    
    #plot line at z=0
    ylim = len(series)
    ax.plot([0, 0], [-1, ylim+1], [0, 0],color = 'black')   # extend in y direction
    ax.set_ylim(0,ylim)
    
    fig.set_facecolor('white')
    ax.set_facecolor('white') 
    ax.grid(color='white', linestyle='-', linewidth=0.5, alpha=0.9)
    
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

def plot_3d_series(df,colors = None):
    
    #create figure and axes
    fig = plt.figure(figsize=(12, 12))
    ax = plt.subplot(111, frameon=False)
    
    data = np.transpose(df.values)
    X = np.linspace(-1, 1, data.shape[-1])
    #G = 1.5 * np.exp(-4 * X ** 2)
    
    lines = []
    for i in range(data.shape[0]):
        # Reduction of the X extents to get a cheap perspective effect
        xscale = 1 - i /100.
        # Same for linewidth (thicker strokes on bottom)
        lw = 1.5 - i / 100.0
        line, = ax.plot(xscale * X, i + data[i], color=colors[i], lw=lw)
        

if __name__ == '__main__':
    
    plt.close('all')
    
    subs = ['Copper','Methomyl','Verapamil','Zinc']
    sub_colors = {
        'Copper':'#d62728',
        'Methomyl':'#9467bd',
        'Verapamil':'#8c564b',
        'Zinc':'#e377c2'
            }
    colors = []
    
    values = np.linspace(0,2.5,60)
    plt.figure()
    plt.plot(values)
    plt.plot(powerlimit_scale(values),color = 'orange')
    
    specie = {'E':'Erpobdella','G':'Gammarus','R':'Radix'}
    root_dir = r'D:\VP\ARTICLE2\ArticleData'
    
    #read selected directories
    dfs = {
        'E':pd.read_csv(r'{}\E_X_i_data.csv'.format(root_dir)),
        'G':pd.read_csv(r'{}\G_Y_i_data.csv'.format(root_dir)),
        'R':pd.read_csv(r'{}\R_Z_i_data.csv'.format(root_dir))
        }
    
    #In branch powerscaling - scaling is not yet true
    scaled = False
    for s in dfs:
        df = dfs[s]
        for col in df.columns:
            df[col] = powerlimit_scale(df[col])
            color = sub_colors[str_remove_num(col)]
            colors.append(color)
            
        #plot_3d_time_series_matplotlib([df[col] for col in df.columns],colors)
        plot_3d_series(df,colors)
        
    #repeat with bspline approximation...