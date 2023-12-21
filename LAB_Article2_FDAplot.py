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

if __name__ == "__main__":
    
    #simulate data
    substances = ['Copper','Methomyl','Verapamil','Zinc']
    colors = ['#8c564b','#e377c2','#bcbd22','#17becf']
    counts = {s:3 for s in substances}
    plot_colors = dict(zip(substances,colors))
    custom_palette = [plot_colors[s] for s in plot_colors]
    sns.set_palette(custom_palette)
    
    y = ['Copper','Copper','Copper','Methomyl','Methomyl','Methomyl','Verapamil','Verapamil','Verapamil','Zinc','Zinc','Zinc',]
    scores = np.array([
        [0.5,0.5],[0.55,0.62],[0.49,0.57],
        [0.1,-0.6],[0.04,-0.43],[0.0,-0.71],
        [-0.2,0.4],[-0.5,0.3],[-0.32,0.37],
        [0.4,-0.2],[0.45,-0.17],[0.41,-0.21]])
    
    sub_scores = {}
    for i,s in enumerate(counts):
        sub_scores.update({s:scores[3*i:3*(i+1)]})
    
    fig,ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(x=scores[:, 0], y=scores[:, 1],hue = y,ax = ax)
    ax.set_xlabel("fPC 1 score")
    ax.set_ylabel("fPC 2 score")
    ax.set_title("FPCA scores Gammarus means")
    
    ax.tick_params(labelsize = 13)
    
    fig,ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(x=np.zeros(scores.shape[0]), y=scores[:, 0],hue = y,ax = ax)
    ax.set_title("FPC1 scores")
    ax.tick_params(labelsize = 13)
    
    fig,ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(x=np.zeros(scores.shape[0]), y=scores[:, 1],hue = y,ax = ax)
    ax.set_title("FPC2 scores")
    ax.tick_params(labelsize = 13)
    
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
    fig,ax = plt.subplots(figsize = (11,7))
    ax.plot(grid_points,data_points[0],color = 'black')
    ax.plot(grid_points,data_points[1],color = 'red')
    ax.legend(labels=['Component 1 - {:.2f}'.format(variances[0]),
                      'Component 2 - {:.2f}'.format(variances[1])],
             fontsize = 17)
    ax.set_xlabel('Time (minutes)',fontsize = 16)
    ax.set_ylabel('FPC Score',fontsize = 16)
    ax.set_title('Functional Principal Component plot',fontsize = 18)