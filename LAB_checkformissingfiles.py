# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 10:42:47 2023

Identify deleted files

@author: George
"""

import os
import pandas as pd

file = r'D:\VP\Viewpoint_data\code\Database_dev\allfiles.txt'
files = pd.read_csv(file,names = ['dir','Tox'])

for i in range(files.shape[0]):
    Tox = files['Tox'].iloc[i]
    direc = files['dir'].iloc[i]
    if not os.path.isdir(r'I:\TXM{}-PC\{}'.format(Tox,direc)):
        print(files.iloc[i])