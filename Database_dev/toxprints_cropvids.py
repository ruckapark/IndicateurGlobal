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

crop_dir = r'C:\Users\George\Documents\PythonScripts\Imagetext'
root = os.getcwd()
os.chdir(crop_dir)
import crop_videos as crop
os.chdir(root)

def str_tolist(series):
    """ 
    Column read of strings in following format
    "['string1' 'string2']"
    output : [str1,str2]
    """
    series = series.str.replace("'","")
    return series.str[1:-1].str.split(' ')

def crop_videos(dopage,crop_hours = 48,directory = None,delete = False):
    """
    Use crop_vid to crop all videos to 48 hours beyond 'dopage'
    """
    
    base_ = os.getcwd()
    if not directory: directory = os.getcwd()
    os.chdir(directory)
    vids = [f for f in os.listdir() if '.avi' in f]
    print(vids)
    ffmpeg = r'C:\Users\George\Documents\PythonScripts\Imagetext\ffmpeg.exe'
    
    for vid in vids:
        vid = r'{}\{}'.format(directory,vid)
        start,end = crop.extract_endpoints(vid,ffm_path = ffmpeg,output_dir = directory)
        start,end = crop.get_datetime(start),crop.get_datetime(end)
        
        if (start > dopage) or (end < dopage + datetime.timedelta(hours = crop_hours)):
            print('SKIPPING vid too short!')
            continue
        vid_end = dopage + datetime.timedelta(hours = crop_hours)
        vid_out = crop.crop_vid(vid,start,vid_end,ffm_path = ffmpeg,output_dir = directory)
        if delete:
            if crop.check_samevid(vid,vid_out,ffm_path = ffmpeg,output_dir = directory):
                os.remove(vid)
                os.rename(vid_out,vid) #replace original file with crop
            else:
                print('Video crop unsuccessful.\n Check files.')
        
        os.chdir(base_)
        

if __name__ == '__main__':
    
    #test version
    reg = pd.read_csv('extended_reg.csv')
    reg['root'] = str_tolist(reg['root'])
    reg['End'] = pd.to_datetime(reg['End'],format = '%Y-%m-%d %H:%M:%S')
    
    reg = reg[reg['TxM']<762]
    reg = reg[reg['End']>datetime.datetime(year = 2022,month = 7,day = 25)]
    
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
        root = r'C:\Users\George\Documents\TestVids\TxM{}-PC\{}'.format(int(entry['TxM']),rootdate.strftime('%Y%m%d-%H%M%S'))
        
        #crop videos to 48 hours beyond dopage
        crop_videos(dopage,crop_hours = 24,directory = root,delete = True)
    
    """
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
    """