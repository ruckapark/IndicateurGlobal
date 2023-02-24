# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 12:10:40 2023

Use extended db to free up some space in NAS, by shortening video files to 48 hours after the 

Don't run this while Core2 is running...

@author: George
"""

import pandas as pd
import numpy as np
import os
import datetime

def str_tolist(series):
    """ 
    Column read of strings in following format
    "['string1' 'string2']"
    output : [str1,str2]
    """
    series = series.str.replace("'","")
    return series.str[1:-1].str.split(' ')

if __name__ == '__main__':
    
    #read in extended reg
    reg = pd.read_csv('extended_reg.csv')
    reg['root'] = str_tolist(reg['root'])
    reg['End'] = pd.to_datetime(reg['End'],format = '%Y-%m-%d %H:%M:%S')
    
    #for each entry, find file with start date earlier than dopage (most recent)
    for i in range(reg.shape[0]):
        
        entry = reg.iloc[i]
        dopage = entry['End']
        
        #this will not work for replayed files...
        try:
            datetimes = [datetime.datetime.strptime(t,'%Y%m%d-%H%M%S') for t in entry['root']]
        except:
            continue
        
        #locate video and apply function start end end extract
        datetimes = [d for d in datetimes if d < dopage]
        rootdate = datetimes[-1]
        root = r'I:\TxM{}-PC\{}'.format(int(entry['TxM']),rootdate.strftime('%Y%m%d-%H%M%S'))
        
        #locate videos in I drives
        os.chdir(root)
        vids = [f for f in os.listdir() if '.avi' in f]
        
        #??
        """
        for v in vids:
            
            extract_vids(v)
            crop_vid(v)
            
            #manually delete other videos for now
        """