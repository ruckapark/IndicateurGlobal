# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 16:04:17 2021

Meausure of léthargie in aquariums reference to pipe change

AVANT_1 = one week before change
APRES_0 = week of change
APRES_1 = week of change

@author: Admin
"""

import dataread as d_
import os

"""Read file and merge"""

#chdir
root = r'D:\VP\Viewpoint_data\Lethargie'
direct = 'AVANT_1'
os.chdir(root)
os.chdir(direct)

#store data file in list (in case multiple?)
tests = []

#check if dir in dir
files = os.listdir()
if os.path.isdir(files[0]):
    
    print('Mulitple tests')
    for direc in files:
        
        pass
    
tests = [d_.read_merge(files,datechange = False)]
for test in tests:
    dfs = d_.preproc(test)
    
    #plot means for gammares, erpo, radix
    for species in 'E G R'.split():
        df = dfs[species]
        df = df.drop(columns = d_.remove_dead(df,species))


#plot individual plots


#think about conditions for removing individuals with the least movement?


#define moving mean that removes most noise 


#look at when the gradient of said curve hits zero