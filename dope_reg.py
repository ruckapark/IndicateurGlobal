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
import sys
import pandas as pd


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

    f = r'D:\VP\Viewpoint_data\code\dope_reg.csv'
    fields = [arg1,arg2,arg3,arg4,arg5,arg6]
    
    with open(f,'a',newline = '') as reg:
        writer = csv.writer(reg, delimiter = '\t')
        writer.writerow(fields)
        
    print('Write file:')
    print('TxM: {}'.format(arg1))
    print('Substance: {} - {}'.format(arg1,arg2))
    print('Date: {}'.format(arg5))
        
        
def dope_read():
    
    df = pd.read_csv(r'D:\VP\Viewpoint_data\code\dope_reg.csv',delimiter = '\t',header = None)
    cols = ['TxM','Substance','Molecule','Concentration','Start','End']
    df.columns = cols
    df['Start'] = pd.to_datetime(df['Start'],format = '%d/%m/%Y %H:%M:%S')
    df['End'] = pd.to_datetime(df['End'],format = '%d/%m/%Y %H:%M:%S')
    df = df.sort_values(by = ['Start','TxM'])
    return df