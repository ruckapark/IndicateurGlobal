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
    
    addfiles = []
    
    for Tox in range(760,770):
        
        basedir = r'I:\TXM{}-PC'.format(Tox)
        os.chdir(basedir)
        dirs = [d for d in os.listdir() if os.path.isdir(d)]
        for d in dirs:
            files = os.listdir(d)
            replayxls = [f for f in files if 'replay.xls' in f]
            xls = [f for f in files if '.xls' in f]
            if len(replayxls) == 1:
                directory = r'{}\{}'.format(basedir,d)
                
                #Check if already treated
                if len([f for f in files if 'Gammarus.csv.zip' in f]):
                    continue
                
                if len(xls) == 2:
                    #add to list
                    addfiles.append(directory)
                else:
                    print('Check dir: ',directory)
                    
    #add all files
    addfile = r'D:\VP\Viewpoint_data\replaydata.txt'
    if os.path.isfile(addfile):
        os.remove(addfile)
    file = open(addfile,'w')
    for item in addfiles:
    	file.write(item+"\n")
    file.close()