# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 22:24:12 2021

@author: Admin
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

protocole = ['1L forage']*5
protocole.extend(['0.5L forage']*3)
protocole.extend(['0.25L forage'])
protocole.extend(['Evian'])
protocole.extend(['1L forage bullee']*3)
protocole.extend(['Retour radix']*5)
protocole.extend(['Retour radix effet - Radix']*2)
protocole.extend(['Retour Gammarus']*2)
protocole.extend(['1L forage froide (choc)']*2)
protocole.extend(['1L becher']*2)
protocole.extend(['0.25L seringue'])

results = np.array([10,11,14,10,6,
           11,4,3,
           3,
           15,
           5,15,9,
           2,0,0,0,3,
           6,2,
           1,15,
           13,2,
           6,1,
           1])/16

result = pd.DataFrame({'Protocole':protocole,'Percentage changepoint detection':results})


fig = plt.figure()
axe = fig.add_axes([0.1,0.3,0.8,0.6])
axe.tick_params(axis = 'x',labelrotation = 90)
sns.boxplot(x = 'Protocole',y = 'Percentage changepoint detection',data = result,color = 'skyblue',whis = 100)
axe.axhline(0.3,color = 'r')