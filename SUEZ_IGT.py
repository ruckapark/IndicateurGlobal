# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 16:44:49 2021

Read terrain data

@author: Admin
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import dataread_terrain as d_terr
from datetime import timedelta
from data_merge import merge_dfs

plt.close('all')


### Functions

def weight_parameters(x,mean,p0):
    
    """ 
    Fit for exponential curve
    """
    
    from scipy.optimize import curve_fit
    params = curve_fit(lambda t,a,b: a*np.exp(b*t), x, mean, p0)
    
    A = params[0][0]
    b = params[0][1]
    
    return A,b
    
def predict(x,A,b):
    """ Return approx from above function"""
    return A*np.exp(b*x)


### Main

if __name__ == '__main__':

    colors = [
        '#42f5e0','#12aefc','#1612fc','#6a00a3',
        '#8ef743','#3c8f01','#0a4001','#fc03ca',
        '#d9d200','#d96c00','#942c00','#fc2803',
        '#e089b6','#a3a3a3','#7a7a7a','#303030'
        ]
    
    root = r'D:\VP\Viewpoint_data\Suez'
    os.chdir(root)
    files = [f for f in os.listdir() if '.csv' in f]
    
    print('The following files will be studied:')
    print(files)
    
    dfs,dfs_mean = d_terr.read_data_terrain(files)
    
    
    timestep = 10
    
    species = 'G'
    df_dist = dfs[species]
    df_dist_mean = dfs_mean[species]
    
    
    # plot all on same figure - no mean and mean
    d_terr.single_plot(df_dist,species,title = 'Distance covered')
    d_terr.single_plot(df_dist_mean,species,title = 'Distance covered movingmean')
    
    # plot individually (plot16)
    fig,axe = d_terr.plot_16(df_dist_mean)    
        
    # IGT
    fig_q, axe_q = plt.subplots(3,3,sharex = True,figsize = (15,10))
    for i in range(9):
        quant = (i+1)*0.05
        IGT = df_dist.quantile(q = quant, axis = 1)**2
        axe_q[i//3,i%3].plot(IGT.index,IGT,color = colors[i])
        
    # means
    plt.figure()
    plt.plot(df_dist.index,df_dist.mean(axis = 1))
    
    
    #%% means - have a look at a distribution
    from matplotlib import animation
    fig = plt.figure()
    ax = fig.add_axes([0.1,0.1,0.8,0.8],xlim = (0,15),ylim = (0,300))
    line, = ax.plot([],[])
    x = np.linspace(0,len(df_dist.columns)-1,len(df_dist.columns))
    
    def init():
        line.set_data([],[])
        return line,
    
    def animate(i):
        y = df_dist.iloc[i].sort_values()
        line.set_data(x,y)
        return line,
    
    
    anim = animation.FuncAnimation(fig, animate, init_func = init, frames = len(df_dist)-1, interval = 30, blit = True)
    plt.show()
    
    
    #find the distribution of all the columns
    values = np.array(df_dist)
    values.sort()
    for i in range(len(values[0])):
        plt.figure()
        sns.histplot(values[:,i])
    
    #find the distribution of the means of the sorted column (nice exponential curve)
    plt.figure()
    mean_distribution = np.mean(values,axis = 0)
    x_range = np.linspace(0,15,16)
    plt.plot(x_range,mean_distribution)
    
    #Aexp(b*x)
    A,b = weight_parameters(x_range,mean_distribution,p0 = (4,0.5))
    
    pred = predict(x_range,A,b)
    
    #adjust to start at 0
    morts = np.linspace(0,15,16,dtype = int)
    weight_coeff = {m:predict(np.linspace(0,15,16-m),A,b) for m in morts}
    
    plt.plot(x_range,pred,color = 'red')
    
    #try to use it as an inverse weighting curve
    #coeffs = 2* pred[::-1]/pred[::-1][0]
    
    coeffs = 2 / ((mean_distribution + 2))
    
    #square to nothing
    IGT_new = np.sum(values**coeffs,axis = 1)
    
    #plot the new IGT
    plt.figure()
    plt.plot(df_dist.index,IGT_new)
    
    
    # find a linear interpolation, depending on how many organisms are alive (dictionary)