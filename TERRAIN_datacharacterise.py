# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 15:21:28 2022

File to describe in detail the data of each species for a terrain data file

Depends on : TERRAIN_readdata.py & dataread_terrain.py

@author: GD0llo
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% IMPORT personal mods
os.chdir('MODS')
import dataread_terrain as d_terr
os.chdir('..')
import TERRAIN_readdata as d_

if __name__ == '__main__':
    
    # Read in data file qnd run readdata to obtain data files and form basic plots
    path = r'D:\VP\Viewpoint_data\TERRAIN\AltenRhein776'
    file = ['toxmate_051121-191121.csv']
    data = d_.main(file,'EGR',root = path)
    
    #%% Inspect data
    
    #check IGT calculation with raw values for 10 values
    
    
    #Check mortality values correspond
    
    
    #Check what happens
    