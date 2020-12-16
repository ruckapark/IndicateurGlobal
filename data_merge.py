"""
Make all dataframes one.
Method:
    - Assume last file is when the doping occured (this can be checked)
    - Read in all xls files in dataframe
    - Remove unwanted values from df (same number of points for all divisible by 48)
    - Remove unwanted columns
    - Take first date of last df and last date of penultimate df
    - find time difference
    - add (time_diff - 20) to all values
    - merge two dfs
    - repeat process if more than two df originally
"""

import os
import pandas as pd

def merge_dfs(dfs):
    """
    merges a list of TWO dfs, - this is thus made recursive
    """
    
    merge_date = dfs[-1]['time'].iloc[0]
    end_date_toshift = dfs[-2]['time'].iloc[-1]
    end_date_shifted = merge_date - pd.Timedelta(seconds = 20)
    time_delta = end_date_shifted - end_date_toshift
    
    dfs[-2]['time'] = dfs[-2]['time'] + time_delta
    df = pd.concat(dfs[-2:])
    df = df.reset_index(drop = True)
    
    if len(dfs) > 2:
        dfs.pop(-1)
        dfs.pop(-2)
        dfs.append(df)
        
        merge_dfs(dfs)
    else:
        return df


#os.chdir(r'D:\VP\Viewpoint_data\TxM{}-PC'.format(Tox))
os.chdir(r'D:\VP\Viewpoint_data\TxM763-PC')
files = os.listdir()

print('The following files will be merged:')
print(files)

dfs = []
for file in files:
    df = pd.read_csv(file,sep = '\t',encoding = 'utf-16')    #read each df in directory df
    df = df[df['datatype'] == 'Locomotion']                         #store only locomotion information
    maxrows = len(df)//48
    df = df.iloc[:maxrows*48]

    #sort values sn = , pn = ,location = E01-16 etcc., aname = A01-04,B01-04 etc.
    df = df.sort_values(by = ['sn','pn','location','aname'])
    df = df.reset_index(drop = True)

    #treat time variable - this gets the days and months the wrong way round
    df['time'] = pd.to_datetime(df['stdate'] + " " + df['sttime'], format = '%d/%m/%Y %H:%M:%S')
    dfs.append(df)
        
  
df = merge_dfs(dfs)