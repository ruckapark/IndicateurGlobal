# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 11:36:44 2023

@author: George
"""

import numpy as np
import matplotlib.pyplot as plt

plt.figure()
for i in range(9,17):
    spacing = np.linspace(0,100,i)
    for x in spacing:
        plt.vlines(x, i, i+1)
       
optimum0 = np.linspace(0,100,16)[1]
optimum = np.linspace(0,100,9)[1] + ((np.linspace(0,100,16)[2] - np.linspace(0,100,9)[1])/2)
optimum2 = np.linspace(0,100,12)[2] + ((np.linspace(0,100,16)[3] - np.linspace(0,100,12)[2])/2)

plt.axvline(optimum0,color = 'orange',linestyle = '--')
plt.axvline(100 - optimum0,color = 'orange',linestyle = '--')

plt.axvline(optimum,color = 'r')
plt.axvline(100 - optimum,color = 'r')

plt.axvline(optimum2,color = 'r',linestyle = '--')
plt.axvline(100 - optimum2,color = 'r',linestyle = '--')