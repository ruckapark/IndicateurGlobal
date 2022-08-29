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

def find_reaction(df,dopage = 0):
    if dopage:
        delta = pd.Timedelta(hours = 1)
    else:
        delta = 1
    reaction_period = df[df.index > dopage + delta]
    try:
        return reaction_period[reaction_period > 100].index[0]
    except:
        return None


specie = {'E': 'Erpobdella','G':'Gammarus','R':'Radix'}

#%% UNWANTED FILES FOR DOSE RESPONSE
unwanted_files = ['760_Methomyl3.xls','760_Methomyl5.xls','761_Methomyl4.xls','762_Methomyl1.xls','762_Methomyl2.xls','762_Methomyl3.xls']

#%% code
if __name__ == "__main__":
    
    directory = r'D:\VP\ARTICLE1_methomyl\Data' #methomyl or copper
    substance = 'meth' #meth or copper
    
    #Article1 data
    os.chdir(directory)
    
    #compressed files have " _ " in file name
    files = [f for f in os.listdir() if (('_' in f) and (f not in unwanted_files))]
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
        
        reaction = find_reaction(quantile,dopage)
        
        if conc == 0:
            quantile_analysis = quantile[(quantile.index > dopage) & (quantile.index < dopage + pd.Timedelta(hours = 3))]
            result.loc[f]['max'] = np.max(quantile_analysis)
            result.loc[f]['ratio'] = find_IGTratio(quantile,dopage)
            result.loc[f]['int'] = np.trapz(quantile_analysis)
            result.loc[f]['conc'] = conc
        elif reaction:
            quantile_analysis = quantile[(quantile.index > reaction) & (quantile.index < reaction + pd.Timedelta(hours = 3))]
            result.loc[f]['max'] = np.max(quantile_analysis)
            result.loc[f]['ratio'] = find_IGTratio(quantile,dopage)
            result.loc[f]['int'] = np.trapz(quantile_analysis)
            result.loc[f]['conc'] = conc
        else:
            #remove file from df
            result = result.drop(f)
    
    measures = list(result.columns)
    measures.remove('conc')
    for m in measures:
        plt.figure()
        sns.boxplot(x = 'conc',y = m,data = result)
        plt.yscale('log')
        plt.title('Dose Response by {}'.format(m))
        plt.xlabel('Concentration $(\mu gL^{-1})$')
        
    #article figure
    if substance == 'meth':
        s = 'Methomyl'
    else:
        s = 'Copper'
    
    fig = plt.figure(figsize = (6,6))
    axe = fig.add_axes([0.15,0.1,0.8,0.8])
    sns.boxplot(x = 'conc',y = 'int',data = result,ax = axe)
    
    #label number of values
    ns = result['conc'].value_counts().sort_index()
    pos = {
        'copper':[0.07e7,0.25e7,0.8e7,2.07e7],
        'meth':[0,0,0,0,0]
        }
    for i in range(len(ns)):
        axe.text((i-0.1),pos[substance][i],'n = {}'.format(ns.iloc[i]))
    
    #axe.set_yscale('log')
    axe.set_title('Dose Response for {}'.format(s),fontsize = 20)
    axe.set_xlabel('Concentration $(\mu gL^{-1})$',fontsize = 18)
    axe.set_ylabel('Total Avoidance $(mm)$',fontsize = 18)
    
    #fig.savefig(r'C:\Users\Admin\Documents\Viewpoint\Article1\{}_{}'.format('Fig2B',s))