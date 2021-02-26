# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 17:34:29 2021

@author: Admin
"""


import os
import pandas as pd
import numpy as np



os.chdir(r'D:\VP\Viewpoint_data\Suez_res')
files = [file for file in os.listdir() if '.csv' in file]

for file in files:
    
    if 'G' in file:
        spec = 'Gammarus'
    elif 'R' in file:
        spec = 'Radix'
    elif 'E' in file:
        spec = 'Erpobdella'
    else:
        spec = None
        
    filename = file.split('_')[0] + spec + '.txt'
        
    data = pd.read_csv(file)
    data['time'] = pd.to_datetime(data['time'], format='%Y%m%d %H:%M:%S').astype(np.int64)
    
    with open(filename, 'w') as f:
        
        for i in range(len(data)):
            f.write('f5683285-b5fa-4be0-be99-6e20a112fad5,sensor={} toxicityindex={} {}\n'.format(spec,data.iloc[i]['toxicityindex'],int(data.iloc[i]['time'])))