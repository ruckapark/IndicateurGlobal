# -*- coding: utf-8 -*-
"""
Created on Sun May 30 23:04:41 2021

Join text files with unix line endings

@author: Admin
"""

import shutil
import os

os.chdir(r'D:\VP\Viewpoint_data\Suez_res')

with open('output_file.txt','wb') as wfd:
    for f in os.listdir():
        with open(f,'rb') as fd:
            shutil.copyfileobj(fd, wfd)