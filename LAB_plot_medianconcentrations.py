# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 12:27:51 2022

@author: George
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

copper_concentrations = np.array([0.0268,0.0928,0.257,0.102,
                         0.0966,0.146,0.036,0.341,0.266,
                         0.207,0.055,0.0178,0.0213,0.0406,
                         0.0259,0.03,0.0025,0.0532,0.0043,
                         0.0289,0.0193,0.0294,0.17,0.208,0.178,
                         0.352,0.634,0.102,0.0789,0.0145,0.00928,
                         0.024,1.0,0.98587895,1.0,1.0,1.0,0.182]) * 1000  #ug/L

fig = plt.figure(figsize = (5,10))
ax = fig.add_subplot(111)

ax.set_yscale('log',basey = 2)
ax.set_yticks([2,20,100,500,1000])
ax.get_yaxis().set_major_formatter(ScalarFormatter())
ax.set_ylim([1., 1500.])

ax.set_xlabel('Copper Chloride')

ax.set_title('Median - EC50 Cuivre Daphnia Magna')
ax.set_ylabel('Concentration {}'.format(r'$\mu g / L$'))

ax.boxplot(copper_concentrations)
ax.set_xticks([])