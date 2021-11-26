# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 08:32:47 2020

@author: George
"""

"""
Development of function to correct xls file for a corrupt output date - if using the RAW or phr files, this will not work.

Function should be added to dataread
"""

import pandas as pd

# FUNCTIONS

def convert_date(filename):
    """
    format 20201230-093312.xls becomes dt(dd30/mm12/YYYY2020) and dt(09:33:12) 
    """
    day_unformatted = filename.split('-')[0]
    day_formatted = day_unformatted[:4] + '/' + day_unformatted[4:6] + '/' + day_unformatted[6:]
    
    if len(filename.split('-')) > 2:
        hour_unformatted = filename.split('-')[1]    
    else:
        hour_unformatted = filename.split('-')[1].split('.')[0]
    hour_formatted = hour_unformatted[:2] + ':' + hour_unformatted[2:4] + ':' + hour_unformatted[4:]
        
    date = day_formatted + ' ' + hour_formatted
    
    return pd.to_datetime(date, format = "%Y/%m/%d %H:%M:%S")

#two system inputs argv[1] and argv[2]

#file = r'{}'.format(sys.argv[1])
#file = r'D:\VP\Viewpoint_data\TxM767-PC\20201204-100452.xls'
 
def correct_dates(file):    

    #extract date from target_file (it is a .xls but should be read as csv)
    true_date = convert_date(file.split('\\')[-1])
    
    #read in data
    df = pd.read_csv(file,sep = '\t',encoding = 'utf-16')
    
    #make new np vector/array of all the combines datetimes (lambda function)
    df['time'] = pd.to_datetime(df['stdate'] + " " + df['sttime'], format = '%d/%m/%Y %H:%M:%S')
    
    #redo dates
    false_date = df['time'].min()
    if false_date < true_date:
        diff = true_date - false_date
        df['time'] = df['time'] + diff
    else:
        diff = false_date - true_date
        df['time'] = df['time'] - diff
    
    # from time column, rewrite 'stdate' and 'sttime'
    df['stdate'] = df['time'].dt.strftime('%d/%m/%Y')
    df['sttime'] = df['time'].dt.strftime('%H:%M:%S')
        
    # delete time column
    df = df.drop('time',1)
    
    #make new_file (add copy at end without deleting original first)
    df.to_csv(file.split('.')[0] + '-copy.xls', sep = '\t', encoding = 'utf-16')