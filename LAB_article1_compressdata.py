# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 09:38:51 2022

Data file to reduce file size around the dopage:
    - read in file and find dopage in meth_reg.csv
    - Delete all rows before dopage - 2h
    - Delete all rows after dopage + 6h

@author: George
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from datetime import timedelta

os.chdir('MODS')
from data_merge import merge_dfs
from dope_reg import dope_read
#import gen_pdf as pdf
import dataread as d_
os.chdir('..')

specie = {'E': 'Erpobdella','G':'Gammarus','R':'Radix'}

def read_data(file):
    
    df = pd.read_csv(file,sep = '\t',encoding = 'utf-16')

    #treat time variable - this gets the days and months the wrong way round
    try:
        df['time'] = pd.to_datetime(df['stdate'] + " " + df['sttime'], format = '%d/%m/%Y %H:%M:%S')
    except ValueError:
        try:
            df['time'] = pd.to_datetime(df['stdate'] + " " + df['sttime'], format = '%Y-%m-%d %H:%M:%S')
        except ValueError:
            df['time'] = pd.to_datetime(df['stdate'] + " " + df['sttime'], format = '%m/%d/%Y %H:%M:%S')
            
    return df

def write_data(df,filename):
    df.to_csv(filename,sep = '\t',encoding = 'utf-16',index = False)

if __name__ == "__main__":
    
    reg = 'meth_reg' #copper_reg meth_reg
    directory = r'D:\VP\ARTICLE1_methomyl\Data' #methomyl or copper
    
    dope_df = dope_read(reg)
    os.chdir(directory)
    
    #test on data file 1
    files = os.listdir()
    
    for file in files:
        
        #read data
        original_df = read_data(file)
        df = d_.read_merge([file])
        dfs = d_.preproc(df)
        
        #TxM number and dopage time for file
        Tox = int(file[:3])
        dopage,date_range,conc,sub,molecule,etude_ = d_.dope_params(dope_df,Tox,dfs[[*dfs][1]].index[0],dfs[[*dfs][1]].index[-1])
        if type(dopage) != pd.Timestamp :
            dopage = dopage.iloc[0]
        
        #find dopage in df
        if 'meth' in reg:
            original_df = original_df[original_df['time'] > (dopage - pd.Timedelta(hours = 2))]
            original_df = original_df[original_df['time'] < (dopage + pd.Timedelta(hours = 10))]
        else:
            original_df = original_df[original_df['time'] > (dopage - pd.Timedelta(hours = 2))]
            original_df = original_df[original_df['time'] < (dopage + pd.Timedelta(hours = 6))]
        
        write_data(original_df.drop(columns = ['time']),'{}_{}.xls'.format(Tox,file.split('.')[0][3:]))