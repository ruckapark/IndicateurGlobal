# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 18:38:49 2023

Plot in style of vertical axis

@author: George
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.spatial import ConvexHull, convex_hull_plot_2d

def ellipse():
    return None
    
def vertical_PC_plot():
    
    return None

def plot_functional_components(grid_points,data_points,variances):
    """
    Parameters
    ----------
    grid_points : np.array
        xvalues - time domain
    data_points : list of np.arrays
        yvalues - eigenvalues forming principal functions
    variances : np.array
        explained variance by each mode of variation

    Returns
    -------
    plot figure and ax
    """
    
    fig,ax = plt.subplots(figsize = (11,7))
    ax.plot(grid_points,data_points[0],color = 'black')
    ax.plot(grid_points,data_points[1],color = 'black',linestyle = '--')
    ax.legend(labels=['Component 1 - {:.2f}'.format(variances[0]),
                      'Component 2 - {:.2f}'.format(variances[1])],
             fontsize = 17)
    ax.set_xlabel('Time (minutes)',fontsize = 16)
    ax.set_ylabel('FPC Score',fontsize = 16)
    ax.set_title('Functional Principal Component plot',fontsize = 18)
    
    return fig,ax
    
def plot_FPscore_projection(scores,y,fpc = 0):
    
    fig,ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x=np.zeros(scores.shape[0]), y=scores[:, fpc],hue = y,ax = ax)
    ax.set_title("FPC1 scores")
    ax.tick_params(labelsize = 15)
    
    return fig,ax
    

if __name__ == "__main__":
    
    plt.close('all')
    
    #simulate data
    substances = ['Copper','Methomyl','Verapamil','Zinc']
    colors = ['#8c564b','#e377c2','#bcbd22','#17becf']
    counts = {'Copper':8,'Methomyl':7,'Verapamil':3,'Zinc':4}
    plot_colors = dict(zip(substances,colors))
    custom_palette = [plot_colors[s] for s in plot_colors]
    sns.set_palette(custom_palette)
    
    y = ['Copper','Copper','Copper','Copper','Copper','Copper','Copper','Copper','Methomyl','Methomyl','Methomyl','Methomyl','Methomyl','Methomyl','Methomyl','Verapamil','Verapamil','Verapamil','Zinc','Zinc','Zinc','Zinc']
    scores = np.array([[ 11.37119376,   0.15997426],
       [ 15.08089024,  -0.95947348],
       [  9.53604394,  -0.08294543],
       [ 19.93950124,   2.49498265],
       [ 19.90819481,   1.77314143],
       [ 10.47929036,  -0.56761362],
       [ 10.52591876,   0.1539535 ],
       [ 19.24454656,   0.36279863],
       [-15.91122718,  -0.79483587],
       [-13.81208783,  -1.53797418],
       [-13.17572315,   4.89341634],
       [-12.9234747 ,   5.25953403],
       [-14.83382371,   0.52309198],
       [-12.13166241,   2.3186874 ],
       [-15.67174559,  -1.12041467],
       [ -3.39789276,  -2.90115001],
       [ -4.02755272,  -2.59719589],
       [ -1.19270985,  -2.70957585],
       [ -5.09301745,  -1.71847185],
       [ -4.32591445,  -2.44412597],
       [ -0.03749463,   0.04755896],
       [  0.44874673,  -0.55336237]])
    
    sub_scores = {}
    for i,s in enumerate(counts):
        sub_scores.update({s:scores[3*i:3*(i+1)]})
    
    fig,ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=scores[:, 0], y=scores[:, 1],hue = y,ax = ax)
    ax.set_xlabel("fPC 1 score")
    ax.set_ylabel("fPC 2 score")
    ax.set_title("FPCA scores Gammarus means")
    
    ax.tick_params(labelsize = 13)
    
    fig,ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(-1,1)
    sns.scatterplot(x=np.zeros(scores.shape[0]), y=scores[:, 0],hue = y,ax = ax,zorder = 2)
    ax.set_title("FPC1 scores")
    ax.tick_params(labelsize = 13)
    locs = ax.get_yticks()[::2]
    ax.axvline(0,color = 'black',zorder = 1)
    for y_tick in locs:
        ax.plot([-0.03, 0], [y_tick, y_tick], color='black', linewidth=1, zorder=1)
    
    fig,ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(-1,1)
    sns.scatterplot(x=np.zeros(scores.shape[0]), y=scores[:, 1],hue = y,ax = ax,zorder = 2)
    ax.set_title("FPC2 scores")
    ax.tick_params(labelsize = 13)
    locs = ax.get_yticks()[::2]
    ax.axvline(0,color = 'black',zorder = 1)
    for y_tick in locs:
        ax.plot([0, 0.03], [y_tick, y_tick], color='black', linewidth=1, zorder=1)
    
    #example of PC1 and PC2
    #Convert fpc1 and fpc2 to np.arrays
    #grid_points,data_points,variances = get_fp_data(fpcas['E'])
    grid_points = np.array([  0. ,   7.2,  14.4,  21.6,  28.8,  36. ,  43.2,  50.4,  57.6,
        64.8,  72. ,  79.2,  86.4,  93.6, 100.8, 108. , 115.2, 122.4,
       129.6, 136.8, 144. , 151.2, 158.4, 165.6, 172.8, 180. , 187.2,
       194.4, 201.6, 208.8, 216. , 223.2, 230.4, 237.6, 244.8, 252. ,
       259.2, 266.4, 273.6, 280.8, 288. , 295.2, 302.4, 309.6, 316.8,
       324. , 331.2, 338.4, 345.6, 352.8, 360. ])
    data_points = [
        np.array([-0.00945615, -0.00677191, -0.00246564,  0.00312498,  0.00966226,
        0.01680853,  0.02422611,  0.03157732,  0.03852448,  0.04474805,
        0.0501393 ,  0.05472552,  0.05853623,  0.061601  ,  0.06394937,
        0.06561089,  0.06661509,  0.06699222,  0.06681312,  0.06621163,
        0.065327  ,  0.06429849,  0.06326536,  0.06236685,  0.06174222,
        0.06153073,  0.06182754,  0.06255153,  0.06357745,  0.0647801 ,
        0.06603424,  0.06721464,  0.0681961 ,  0.06885338,  0.06906125,
        0.0686945 ,  0.0676279 ,  0.06573623,  0.06289938,  0.05911485,
        0.05449778,  0.04916841,  0.04324699,  0.03685376,  0.03010896,
        0.02313284,  0.01604565,  0.00896762,  0.00201901, -0.00467994,
       -0.011009  ]),
        np.array([ 0.05566545,  0.08197939,  0.10110565,  0.11377675,  0.12072526,
        0.12268372,  0.12038466,  0.11456064,  0.1059442 ,  0.09525509,
        0.08306432,  0.06984695,  0.05607642,  0.04222619,  0.02876971,
        0.01618043,  0.00493181, -0.00450709, -0.01193081, -0.01754242,
       -0.02158015, -0.02428224, -0.02588691, -0.02663239, -0.0267569 ,
       -0.02649869, -0.0260595 , -0.02549522, -0.02482528, -0.02406908,
       -0.02324604, -0.02237557, -0.02147711, -0.02057005, -0.01967382,
       -0.01880783, -0.0179915 , -0.01724424, -0.01658362, -0.0159843 ,
       -0.01537811, -0.01469498, -0.01386487, -0.01281772, -0.01148349,
       -0.00979212, -0.00767355, -0.00505774, -0.00187463,  0.00194583,
        0.00647369])]
    variances = np.array([0.96,0.02])
    
    #plot each component
    fig,ax = plt.subplots(figsize = (8,6))
    ax.plot(grid_points,data_points[0],color = 'black')
    ax.plot(grid_points,data_points[1],color = 'black',linestyle = '--')
    ax.legend(labels=['Component 1 - {:.2f}'.format(variances[0]),
                      'Component 2 - {:.2f}'.format(variances[1])],
             fontsize = 17)
    ax.set_xlabel('Time (minutes)',fontsize = 16)
    ax.set_ylabel('FPC Score',fontsize = 16)
    ax.set_title('Functional Principal Component plot',fontsize = 18)