# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 08:29:30 2020

Write line to csv reg

Read csv reg

This could be made a class - with various functions within read

dope_reg.TxM763 etc.

@author: Admin
"""

import csv
#import sys
import pandas as pd
import numpy as np


def dope_write(arg1,arg2,arg3,arg4,arg5,arg6):
    """
    

    Parameters
    ----------
    arg1 : int
        Toxmate number in form: 763 - useful for sorting
    arg2 : String
        Substance in form 'Zinc'
    arg3 : String
        ChemicalFormula in form 'Zn(SO4)_7(H20)'
        _ represents a dot
        () around ion groups
    arg4 : String
        concentration in form '10ug' assumed per litre
    arg5 : String
        Begin dope time in form: '17/12/2020 09:55:00' seconds always zero
    arg6 : String
        End dope time in form: '17/12/2020 09:58:00' seconds always zero

    Returns
    -------
    Saves row to the doping register - this should be done after each ToxMate doping.

    """

    f = r'D:\VP\Viewpoint_data\code\REGS\dope_reg.csv'
    fields = [arg1,arg2,arg3,arg4,arg5,arg6]
    
    with open(f,'a',newline = '') as reg:
        writer = csv.writer(reg, delimiter = ',')
        writer.writerow(fields)
        
    print('Write file:')
    print('TxM: {}'.format(arg1))
    print('Substance: {} - {}'.format(arg1,arg2))
    print('Date: {}'.format(arg5))
        
        
def dope_read(reg = None):
    
    if reg: 
        df = pd.read_csv(r'D:\VP\Viewpoint_data\code\REGS\{}.csv'.format(reg),delimiter = ',',header = None)
    else:
        df = pd.read_csv(r'D:\VP\Viewpoint_data\code\REGS\dope_reg.csv',delimiter = ',',header = None)
    cols = ['TxM','Substance','Molecule','Concentration','Start','End']
    df.columns = cols
    df['Start'] = pd.to_datetime(df['Start'],format = '%d/%m/%Y %H:%M:%S')
    df['End'] = pd.to_datetime(df['End'],format = '%d/%m/%Y %H:%M:%S')
    df = df.sort_values(by = ['Start','TxM'])
    return df

def str_tolist(series):
    """ 
    Column read of strings in following format
    "['string1' 'string2']"
    output : [str1,str2]
    """
    series = series.str.replace("'","")
    return series.str[1:-1].str.split(' ')

def dope_write_extend(root = r'D:\VP\Viewpoint_data\code\REGS'):
    
    reg = dope_read()
    allfiles = pd.read_csv('allfiles.txt',delimiter = ',',names = ['root','Tox'])
    allfiles['datetime'] = pd.to_datetime(allfiles['root'],format = '%Y%m%d-%H%M%S')
    
    reg['shortfile'] = np.nan
    reg['root'] = np.nan
    for i in range(reg.shape[0]):
        entry = reg.iloc[i]
        Tox = entry['TxM']
        dope = entry['End']
        if dope.weekday() >=4:
            dope_limit = dope - pd.Timedelta(days = dope.weekday())
        else:
            dope_limit = dope - pd.Timedelta(days = 6)
            
        files = allfiles[allfiles['Tox'] == Tox]
        files = files[(files['datetime'] < (dope + pd.Timedelta(hours = 24))) & (files['datetime'] > dope_limit)]
        
        if files.shape[0] == 1:
            reg['shortfile'].iloc[i] = 1
            reg['root'].iloc[i] = [files.iloc[0]['root']]
        else:
            reg['shortfile'].iloc[i] = 0
            reg['root'].iloc[i] = files['root'].values
            
    reg.to_csv(r'{}\extended_reg.csv'.format(root),index = False)



def dope_read_extend():
    
    """ Currently extended reg has alternate format """
    
    reg = pd.read_csv(r'D:\VP\Viewpoint_data\code\REGS\extended_reg.csv')
    reg['root'] = str_tolist(reg['root'])
    reg['End'] = pd.to_datetime(reg['End'],format = '%Y-%m-%d %H:%M:%S')
    return reg


"""

Example of dope_write

dope_write(763,Zinc,Zn(SO4)_7(H20),'10ug,'17/12/2020 09:55:00','17/12/2020 09:55:00')

"""