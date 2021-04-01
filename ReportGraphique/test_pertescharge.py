# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 08:31:40 2021

@author: Admin
"""


# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 22:24:12 2021

@author: Admin
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

protocole = ['1L +- 0cm']*12
protocole.extend(['1L +30cm']*3)
protocole.extend(['1L -15cm']*2)
protocole.extend(['3L dans eau +-0cm']*2)

results = np.array([35.71,36.48,35.11,34.11,35.45,35.44,35.78,36.10,36.14,36.16,35.39,35.99,
                    44.12,44.02,43.97,
                    32.10,32.83,
                    100.71/3,100.85/3
           ])

result = pd.DataFrame({'Protocole':protocole,'Temps fill 1L':results})


fig = plt.figure()
axe = fig.add_axes([0.1,0.3,0.8,0.6])
axe.tick_params(axis = 'x',labelrotation = 90)
sns.boxplot(x = 'Protocole',y = 'Temps fill 1L',data = result,color = 'skyblue',whis = 100)