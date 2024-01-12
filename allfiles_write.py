# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 16:35:40 2024

generate allfiles.txt

@author: George
"""

import os

filename = 'allfiles.txt'

allfiles = [[],[]]
for Tox in range(760,770):
    
    files = os.listdir(r'I:\TXM{}-PC'.format(Tox))
    files = [f for f in files if len(f) == 15]
    Toxs = [Tox]*len(files)
    
    allfiles[0].extend(files)
    allfiles[1].extend(Toxs)
    
#write to file
zipped_data = list(zip(*allfiles))

# Write to file
with open(r'D:\VP\Viewpoint_data\REGS\{}'.format(filename), 'w') as file:
    for line in zipped_data:
        file.write(','.join(map(str, line)) + '\n')