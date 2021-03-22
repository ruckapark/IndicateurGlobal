# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 11:18:29 2021

@author: Admin
"""


import shutil
import os

os.chdir(r'D:\VP\Viewpoint_data\_??')

with open('output.txt','wb') as wfd:
    for f in os.listdir():
        with open(f,'rb') as fd:
            shutil.copyfileobj(fd, wfd)