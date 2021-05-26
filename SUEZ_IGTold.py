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
    
#could have used predict to find the best fit of sigmoid function?
    
#Write IGT to a class
    


def predict(x,A,b):
    """ Return approx from above function"""
    return A*np.exp(b*x)

def sigmoid_coeffs(m,species):
    
    n = 16-m
    
    a = {'G':3.5,'E':3.5,'R':6.0}
    b = {'G':3.0,'E':2.5,'R':2.5}
    
    if n <= 1: 
        return np.array([0])
    elif (n > 1) & (n < 7):
        x = np.arange(2,n+2)
        return 2* (-1/(1+a[species]**(-x+b[species])) + 1) + 0.15
    else:
        x = np.arange(1,n+1)
        return 2* (-1/(1+a[species]**(-x+b[species])) + 1) + 0.15
    
# def percent_IGT(IGT,species):
    
#     """
#     The problem here is the assumption that all the organisms are alive - the conversion to percentage will vary depending on the number alive...
#     """
    
#     if species == 'G':
#         IGT[IGT <= 40] = np.sqrt(IGT[IGT <= 40]) * 1.5812
#         IGT[IGT > 40] = (np.log(IGT[IGT > 40])/np.log(1.5) - np.log(40)/np.log(1.5))*5.5 + 10
#     elif species == 'E':
#         IGT[IGT <= 100] = np.sqrt(IGT[IGT <= 100])
#         IGT[IGT > 100] = (np.log(IGT[IGT > 100])/np.log(1.5) - np.log(100)/np.log(1.5))*5.5 + 10
#     elif species == 'R':
#         IGT[IGT <= 22.5] = np.sqrt(IGT[IGT <= 22.5]) * 2.108
#         IGT[IGT > 22.5] = (np.log(IGT[IGT > 22.5])/np.log(1.5) - np.log(22.5)/np.log(1.5))*5.5 + 10
    
#     IGT_mean = np.ones(len(IGT)//3)
#     for i in range(len(IGT)//3):
#         IGT_mean[i] = np.sum(IGT[3*i:3*i+3])/3
#     return IGT_mean

def percent_IGT(IGT,species):
    
    """
    The problem here is the assumption that all the organisms are alive - the conversion to percentage will vary depending on the number alive...
    """
    
    if species == 'G':
        #IGT[IGT <= 50] = IGT[IGT <= 50] /5
        #IGT[IGT > 50] = (np.log10(IGT[IGT > 50] - 40))*20 + 10
        #IGT[IGT <= 20] = IGT[IGT <= 20] /2
        #IGT[IGT > 20] = (np.log10(IGT[IGT > 20] - 10))*20 + 10
        IGT[IGT <= 40] = IGT[IGT <= 40] /4
        IGT[IGT > 40] = (np.log10(IGT[IGT > 40] - 30))*21 + 9
    elif species == 'E':
        #IGT[IGT <= 100] = IGT[IGT <= 100] / 100
        #IGT[IGT > 100] = (np.log10(IGT[IGT > 100] - 90))*20 + 9
        IGT[IGT <= 40] = IGT[IGT <= 40] / 4
        IGT[IGT > 40] = (np.log10(IGT[IGT > 40] - 30))*21 + 9
    elif species == 'R':
        # IGT[IGT <= 22.5] = np.sqrt(IGT[IGT <= 22.5]) * 2.108
        # IGT[IGT > 22.5] = (np.log(IGT[IGT > 22.5] - 21.5)/np.log(1.5))*3 + 10
        IGT[IGT <= 40] = IGT[IGT <= 40] / 4
        IGT[IGT > 40] = (np.log10(IGT[IGT > 40] - 30))*25 + 9
    
    IGT_mean = np.array(pd.Series(IGT).rolling(12).mean().fillna(0))
    return IGT_mean


def animate_distribution(df):
    
    #mean distribution ordered
    from matplotlib import animation
    fig = plt.figure()
    ax = fig.add_axes([0.1,0.1,0.8,0.8],xlim = (0,15),ylim = (0,300))
    line, = ax.plot([],[])
    x = np.linspace(0,len(df.columns)-1,len(df.columns))
    
    def init():
        line.set_data([],[])
        return line,
    
    def animate(i):
        y = df_dist.iloc[i].sort_values()
        line.set_data(x,y)
        return line,
    
    
    anim = animation.FuncAnimation(fig, animate, init_func = init, frames = len(df_dist)-1, interval = 30, blit = True)
    plt.show()


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
    files = [f for f in os.listdir() if '1003.csv' in f]
    
    print('The following files will be studied:')
    print(files)
    
    dfs,dfs_mean = d_terr.read_data_terrain(files)
    
    
    timestep = 10
    
    species = 'R'
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
    
    
    #find the distribution of all the columns
    values = np.array(df_dist)
    
    values.sort()
    
    # coeffs = 2 / ((mean_distribution + 2))
    coeffs = sigmoid_coeffs(0,species)
    
    #square to nothing
    IGT_new = np.sum(values**coeffs,axis = 1)
    # if species == 'R':
    #     IGT_new[IGT_new > 400] = 5
    #     IGT_new = IGT_new
    
    #plot the new IGT
    plt.figure()
    plt.plot(df_dist.index,IGT_new)
    #plt.yscale('log',basey = 2)
    # find a linear interpolation, depending on how many organisms are alive (dictionary)
    
    
    if species == 'R':
        print('Radix')
        
        IGT_low = IGT_new[IGT_new <= 22.5]
        IGT_high = IGT_new[IGT_new > 22.5]
    elif species == 'G':
        print('Gammarus')
        
        IGT_low = IGT_new[IGT_new <= 40]
        IGT_high = IGT_new[IGT_new > 40]
    elif species == 'E':
        print('Erpobdella')
        
        IGT_low = IGT_new[IGT_new <= 100]
        IGT_high = IGT_new[IGT_new > 100]
        
    IGT = percent_IGT(IGT_new,species)
    plt.figure()
    plt.ylim((0,100))
    plt.plot(IGT,color = 'green')
    plt.fill_between(np.linspace(1,len(IGT),len(IGT)),IGT,color = 'green',alpha = 0.3)
    plt.axhline(y = 50, color = 'orange')
    plt.axhline(y = 75, color = 'red')
    
    
    # treat array for deaths 
    data_alive,data_counters = d_terr.search_dead(np.array(df_dist),species)
    
    values = np.array(df_dist)
    values[data_alive == 0] = np.nan
    values.sort()
    
    # m is list of mortality percentage
    m = np.ones(len(data_alive),dtype = float) - (np.sum(data_alive,axis = 1))/16
    
    IGT = np.zeros_like(m)    
    for i in range(len(values)):
        coeffs = sigmoid_coeffs(int(16*m[i]),species)
        IGT[i] = np.sum(values[i][:len(coeffs)]**coeffs)
        
    # split IGT into high and low
    test = np.array(IGT)
    test.sort()
    plt.figure()
    lim = 100
    sns.distplot(test[(test > 1) & (test < 100)],bins = 100)
    lim = 40
    plt.figure()
    plt.plot(test[test <= lim]/5)
    plt.figure()
    plt.plot(np.log10(test[test > lim] - lim + 10))
       
    
    # caluclate IGT from raw -> %    
    IGT = percent_IGT(IGT, species)
    fig,axe = plt.subplots(1,1,figsize = (18,9))
    axe.plot(df_dist.index,IGT,color = 'green')
    