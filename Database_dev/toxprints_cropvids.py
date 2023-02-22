# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 12:10:40 2023

Use extended db to free up some space in NAS, by shortening video files to 48 hours after the dopage

@author: George
"""

import pandas as pd
import numpy as np
import os
import datetime

if __name__ == '__main__':
    
    #read in extended reg
    reg = pd.read_csv('extended_reg.csv')
    
    #for each entry, find file with start date earlier than dopage (most recent)
    for i in range(reg.shape[0]):
        
        entry = reg.iloc[i]
        dopage = entry['End']
        
        #this will not work for replayed files...
        datetimes = [datetime.datetime(t,'%Y%m%d-%H%M%S') for t in entry['root']]
        
        #locate video and apply function start end end extract
        datetimes = datetimes[datetimes < dopage]
        root = datetimes[-1]
        
        #locate videos in I drives
        os.chdir(root)
        vids = [f for f in os.listdir() if '.avi' in f]
        for v in vids:
            
            extract_vids(v)
            crop_vid(v)
            
            #manually delete other videos for now