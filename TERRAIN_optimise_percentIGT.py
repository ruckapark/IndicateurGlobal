# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 13:20:11 2021

Fit 50 75 percent intervals to IGT for each species

@author: Admin
"""

# optimize fit of log curve per species
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress as lin

## RADIX
paliers_r = np.array([450,600,950,3000])
r2 = []

test = np.linspace(0,449,450,dtype = int)
for c in test:
    temp = paliers_r - c
    r2.append(lin(np.array([1,2,3,4]),np.log(temp))[2])

plt.figure()
plt.plot(r2)

## GAMMARUS
paliers_r = np.array([3500,5000,8000,30000])
r2 = []

test = np.linspace(0,3499,3500,dtype = int)
for c in test:
    temp = paliers_r - c
    r2.append(lin(np.array([1,2,3,4]),np.log(temp))[2])

plt.figure()
plt.plot(r2)

## ERPO
paliers_r = np.array([2500,5000,7000,30000])
r2 = []

test = np.linspace(0,2499,2500,dtype = int)
for c in test:
    temp = paliers_r - c
    r2.append(lin(np.array([1,2,3,4]),np.log(temp))[2])

plt.figure()
plt.plot(r2)