# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 13:53:36 2022

Dose response attempt for Copper

Strat :
    - Get data for all concentrations
    - Divide by 4 (roughly) and exclude outliers (higher thresholds than terrain)
    - Before exclusion - if one organism has too many outlying points remove it completely (more useful for radix)
    - Plot a subplot for each seperate conecentration
    - Give a boxplot for:
        In first three hours
        Peak IGT value
        Time spent above IGT 2500?
    - Start by trying this just for copper
        

@author: George
"""

#%% IMPORTS classic mods
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from datetime import timedelta

#%% IMPORT personal mods
os.chdir('MODS')
from data_merge import merge_dfs
from dope_reg import dope_read
import dataread as d_
os.chdir('..')

#%% Functions
def find_concentrations(dic):
    """ find concentrations in dictionary"""
    concs = []
    for key in dic:
        conc = dic[key][-1]
        if conc not in concs: concs.append(conc)
    concs = np.array(concs)
    concs.sort()
    return concs

def find_conc_data(conc,data,dopages):
    """
    conc : INT
        concentration of doapge in ug/L.
    data : dict
        all data files [df,df_mean] per entry
    dopages : dict
        info on dopage of data files : [dopage,dates,concentration]
    """
    concdata = []
    for key in data:
        if dopages[key][-1] == conc:
            concdata.append(key)
    return concdata

def find_IGTratio(quantile,dope,thresh = 5000):
    
    """ Check how many array entries are above threshold """
    
    quant = np.array(quantile[quantile.index > dope])
    above = len(quant[quant > thresh])
    return above/len(quant)


specie = {'E': 'Erpobdella','G':'Gammarus','R':'Radix'}

#%% code
if __name__ == "__main__":
    
    directory = r'D:\VP\ARTICLE1_copper\Data' #methomyl or copper
    substance = 'copper' #meth or copper
    
    #Article1 data
    os.chdir(directory)
    
    #compressed files have " _ " in file name
    files = [f for f in os.listdir() if '_' in f]
    dope_df = dope_read('{}_reg'.format(substance))
    
    
    #treat each data file, add to dictionary
    data = {}
    dopages = {}
    for file in files:
        
        #Toxname
        Tox = int(file[:3])
        
        #extract only gammarus df
        df = d_.read_merge([file])
        dfs = d_.preproc(df)
        df = dfs['G']
        deadcol = d_.remove_dead(df,'G')
        if len(deadcol) > 14: continue
        df = df.drop(columns = deadcol)
        
        #meaned df
        t_mins = 5
        df_mean = d_.rolling_mean(df,t_mins)
        
        #find dopage time
        dopage,date_range,conc,sub,molecule,etude_ = d_.dope_params(dope_df,Tox,df.index[0],df.index[-1])
        conc = int(conc[:-2])
        
        data.update({file:[df,df_mean]})
        dopages.update({file:[dopage,date_range,conc]})
    
    #%%
    result = pd.DataFrame(index = files,columns = ['max','ratio','int','conc'])
        
    for f in files:
        
        df = data[f][-1]
        mean = df.mean(axis = 1)
        quantile = df.quantile(q = 0.05, axis = 1)**2
        dopage = dopages[f][0]
        conc = dopages[f][-1]
        
        result.loc[f]['max'] = np.max(quantile[quantile.index > dopage])
        result.loc[f]['ratio'] = find_IGTratio(quantile,dopage)
        result.loc[f]['int'] = np.trapz(quantile[quantile.index > dopage])
        result.loc[f]['conc'] = conc
    
    measures = list(result.columns)
    measures.remove('conc')
    for m in measures:
        plt.figure()
        sns.boxplot(x = 'conc',y = m,data = result)
        plt.yscale('log')