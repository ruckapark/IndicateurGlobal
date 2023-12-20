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

def elipse():
    
    return None
    
def vertical_PC_plot():
    
    return None

if __name__ == "__main__":
    
    #simulate data
    plot_colors = {'Copper':'#8c564b','Methomyl':'#9467bd','Verapamil':'#d62728','Zinc':'#e377c2'}
    custom_palette = [plot_colors[s] for s in plot_colors]
    sns.set_palette(custom_palette)
    
    y = ['Copper','Copper','Copper','Methomyl','Methomyl','Methomyl','Verapamil','Verapamil','Verapamil','Zinc','Zinc','Zinc',]
    scores = np.array([
        [0.5,0.5],[0.55,0.62],[0.49,0.57],
        [0.1,-0.6],[0.04,-0.43],[0.0,-0.71],
        [-0.2,0.4],[-0.5,0.3],[-0.32,0.37],
        [0.4,-0.2],[0.45,-0.17],[0.41,-0.21]])
    
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