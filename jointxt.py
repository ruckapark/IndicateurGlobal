# -*- coding: utf-8 -*-
"""
Created on Sun May 30 23:04:41 2021

Join text files with unix line endings

Add header to file.

@author: Admin
"""

import shutil
import os

def jointext(direc,output_name = 'output.txt'):
    
    os.chdir(direc)
    
    #create header
    lines = [
        '# DDL',
        '# DML',
        '# CONTEXT-DATABASE: replaydb',
        '',
        '']
    
    #force first
    with open('!header.txt', 'w', newline = '\n') as f:
        for item in lines:
            f.write('{}\n'.format(item))
            
    with open(output_name,'wb') as wfd:
        for f in os.listdir():
            with open(f,'rb') as fd:
                shutil.copyfileobj(fd, wfd)