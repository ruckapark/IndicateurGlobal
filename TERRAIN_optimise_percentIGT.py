# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 13:20:11 2021

Fit 50 75 percent intervals to IGT for each species
To choose the palier, visualise data (may not be the same in all sites)

@author: Admin
"""

# optimize fit of log curve per species
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress as lin

def find_optimum_offsets(palier = {
        'E':np.array([2500,5000,7000,30000]),
        'G':np.array([3500,5000,8000,30000]),
        'R':np.array([450,600,950,3000])}):
    
    """ 
    Function to find log offset for linear fit in higher range.
    Will default to in function high range fitpoints ('paliers').
    """
    
    offsets = {'E':0,'G':0,'R':0}
    
    for species in offsets:
        paliers = palier[species]
        r2 = []
        
        test = np.linspace(0,paliers[0]-1,paliers[0])
        for c in test:
            temp = paliers - c
            r2.append(lin(np.array([1,2,3,4]),np.log(temp))[2])
            
        offsets[species] = np.argmax(r2)
    
    return offsets

if __name__ == "__main__":

    offset = {'E':0,'G':0,'R':0}
    
    ## RADIX
    sp = 'R'
    paliers_r = np.array([450,600,950,3000])
    r2 = []
    
    test = np.linspace(0,449,450,dtype = int)
    for c in test:
        temp = paliers_r - c
        r2.append(lin(np.array([1,2,3,4]),np.log(temp))[2])
    
    
    plt.figure()
    plt.plot(r2)
    
    offset['R'] = np.argmax(r2)
    
    
    ## GAMMARUS
    sp = 'G'
    paliers_r = np.array([3500,5000,8000,30000])
    r2 = []
    
    test = np.linspace(0,3499,3500,dtype = int)
    for c in test:
        temp = paliers_r - c
        r2.append(lin(np.array([1,2,3,4]),np.log(temp))[2])
    
    plt.figure()
    plt.plot(r2)
    
    offset['G'] = np.argmax(r2)
    
    
    ## ERPO
    sp = 'E'
    paliers_r = np.array([2500,5000,7000,30000])
    r2 = []
    
    test = np.linspace(0,2499,2500,dtype = int)
    for c in test:
        temp = paliers_r - c
        r2.append(lin(np.array([1,2,3,4]),np.log(temp))[2])
    
    plt.figure()
    plt.plot(r2)
    
    offset['E'] = np.argmax(r2)