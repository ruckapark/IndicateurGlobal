# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 14:02:49 2023

- distribution of mortality
- mortality date plot
- mortality per cage

@author: George
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.chdir('MODS')
import dataread as d_
os.chdir('..')

mortality_counts = {'E':[],'G':[],'R':[]}
mortality_bydate = {}
mortality_bycage = {'E':{i:0 for i in range(1,17)},
                    'G':{i:0 for i in range(1,17)},
                    'R':{i:0 for i in range(1,17)}}

specie = {'E':'Erpobdella','G':'Gammarus','R':'Radix'}
colors = {'E':'#db9e0f','G':'#119c36','R':'#3e16de'}
dark = ['#db9e0f','#119c36','#3e16de']

for Tox in range(760,770):
    
    os.chdir(r'I:\TXM{}-PC'.format(Tox))
    
    dirs = [d for d in os.listdir() if os.path.isdir(d)]
    dirs = [d for d in dirs if len(d) == 15]
    
    for d in dirs:
        
        date = d[:8]
        os.chdir(d)
        
        if os.path.isfile('morts.csv'):
            #read morts and stock under date
            morts = d_.read_dead(os.getcwd())
            
            if date not in [*mortality_bydate]:
                mortality_bydate.update({date:{'E':0,'G':0,'R':0}})
            
            for s in morts: 
                mortality_counts[s].append(len(morts[s]))
                if len(morts[s]):
                    for x in morts[s]: mortality_bycage[s][x] += 1
                    mortality_bydate[date][s] += len(morts[s])
            
        os.chdir('..')

#%% Plots   
plt.close('all')
sns.set_style("white")

#plot overall mortality counts
#distribution per species for counts
fig,axe = plt.subplots(1,3,figsize = (16,5))
plt.suptitle('Mortality Count distribution')
for i,s in enumerate(mortality_counts):
    sns.histplot(np.array(mortality_counts[s]),ax=axe[i],color = colors[s],kde = True)
    axe[i].set_title(specie[s])

#plot mortality by date
#loop though dates, if more than 5 in gammarus how to know?

#create appropriate dataframe for grouped barplot
df_mortality_bydate = {'Date':[],'Species':[],'Mortality':[]}
for date in mortality_bydate:
    for s in specie:
        df_mortality_bydate['Date'].append(date)
        df_mortality_bydate['Species'].append(specie[s])
        df_mortality_bydate['Mortality'].append(mortality_bydate[date][s])

df_mortality_bydate = pd.DataFrame(df_mortality_bydate)

g = sns.catplot(
    data=df_mortality_bydate, kind="bar",
    x="Date", y="Mortality", hue="Species",
    palette=dark, alpha=.8
)
g.set_axis_labels("","")
g.legend.set_title("")

#plot mortality by cage for each species
df_mortality_bycage = {'Cage':[],'Species':[],'Mortality':[]}
for s in mortality_bycage:
    for cage in mortality_bycage[s]:
        df_mortality_bycage['Cage'].append(int(cage))
        df_mortality_bycage['Species'].append(specie[s])
        df_mortality_bycage['Mortality'].append(mortality_bycage[s][cage])

df_mortality_bycage = pd.DataFrame(df_mortality_bycage)

g = sns.catplot(
    data=df_mortality_bycage, kind="bar",
    x="Cage", y="Mortality", hue="Species",
    palette=dark, alpha=.8
)
g.set_axis_labels("Cage","")
g.legend.set_title("")