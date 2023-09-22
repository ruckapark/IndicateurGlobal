# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 11:01:13 2023

Code to generate list of all directories for replay data

Condition:
- contains replay xls and xls
- does not already contain replayed file

@author: George
"""

import os

if __name__ == '__main__':
    
    for Tox in range(760,770):
        
        basedir = r'I:\TXM{}-PC'.format(Tox)
        os.chdir(basedir)
        dirs = [d for d in os.listdir() if os.path.isdir(d)]
        for d in dirs:
            files = os.listdir(d)
            replayxls = [f for f in files is 'replay.xls' in files]
            xls = [f for f in files if '.xls' in ]
            if len(replayxls) == 1:
                if len xls == 2:
                    #add to list
                elif len xls == 1:
                    print('Check dir')