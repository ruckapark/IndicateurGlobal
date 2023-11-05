# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 11:26:33 2023

Generate IGT and meandata for dataset2

@author: George
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from datetime import timedelta

#%% IMPORT personal mods
import LAB_ToxClass as TOX
import LAB_splines as SPL

os.chdir('MODS')
from data_merge import merge_dfs
from dope_reg import dope_read_extend
import dataread as d_
os.chdir('..')

#%% data registers
coppers = [
    r'I:\TXM765-PC\20210422-111620',
    r'I:\TXM767-PC\20210430-124553',
    r'I:\TXM767-PC\20210513-231929',
    r'I:\TXM763-PC\20210528-113951',
    r'I:\TXM764-PC\20210409-135918',
    r'I:\TXM767-PC\20210416-113551',
    r'I:\TXM765-PC\20210430-124646',
    r'I:\TXM765-PC\20210506-231620',
    r'I:\TXM767-PC\20210513-231929'
    ]

zincs = [
    r'I:\TXM763-PC\20210416-113757',
    r'I:\TXM763-PC\20210506-230746',
    r'I:\TXM763-PC\20210513-230658',
    r'I:\TXM763-PC\20210520-224858'
    ]

methomyls = [
    r'I:\TXM760-PC\20210520-224501',
    r'I:\TXM760-PC\20210625-093621',
    r'I:\TXM761-PC\20210520-224549',
    r'I:\TXM761-PC\20210625-093641'
    ]

tramadols = [
    r'I:\TXM767-PC\20220225-091008',
    r'I:\TXM768-PC\20220225-090953',
    r'I:\TXM769-PC\20220310-113807',
    r'I:\TXM769-PC\20220317-164759']

#deal with class to only account for active species
datasets = coppers + zincs + methomyls

if __name__ == "__main__":
    
    input_directory = r'D:\VP\ARTICLE2\ArticleData'  #find data means or IGTs
    
    written = True
    
    if not written:
        for r in datasets:
            data = TOX.csvDATA(r)
            data.write_data(r'D:\VP\ARTICLE2\ArticleData')
    
    IGTs = [f for f in os.listdir(input_directory) if 'IGT' in f]
    means = [f for f in os.listdir(input_directory) if 'mean' in f]
    
    dfs_IGT,dfs_mean = ({s:None for s in ['E','G','R']} for i in range(2))
    dfs_IGT_s,dfs_mean_s = ({s:None for s in ['E','G','R']} for i in range(2))
     
    for i in range(len(IGTs)):
        IGT,mean = IGTs[i],means[i]
    
        IGT_s = SPL.ToxSplines(r'{}\{}'.format(input_directory,IGT))
        mean_s = SPL.ToxSplines(r'{}\{}'.format(input_directory,mean))
        
        if not i: 
            for s in dfs_IGT:
                #raw
                dfs_IGT[s] = pd.DataFrame(index = IGT_s.data.index,columns = np.arange(len(IGTs)))
                dfs_mean[s] = pd.DataFrame(index = mean_s.data.index,columns = np.arange(len(means)))
                #smooth
                dfs_IGT_s[s] = pd.DataFrame(index = IGT_s.data.index,columns = np.arange(len(IGTs)))
                dfs_mean_s[s] = pd.DataFrame(index = mean_s.data.index,columns = np.arange(len(means)))
        
        #write data to dataframes
        for s in IGT_s.active_species:
            dfs_IGT[s][i] = IGT_s.data[s].values[:len(dfs_IGT[s].index)]
            dfs_mean[s][i] = mean_s.data[s].values[:len(dfs_IGT[s].index)]
            
            dfs_IGT_s[s][i] = IGT_s.data['{}smooth'.format(s)].values[:len(dfs_IGT_s[s].index)]
            dfs_mean_s[s][i] = mean_s.data['{}smooth'.format(s)].values[:len(dfs_mean_s[s].index)]
        
        IGT_s.plot_raw()
        mean_s.plot_raw()
        
    #write dataframes to csv files - no IGT in titles to avoid clash
    if not written:
        root = r'D:\VP\ARTICLE2\ArticleData'
        dfs_IGT['E'].to_csv('{}\{}_X_i_data.csv'.format(root,'E'),header = False,index = False)
        dfs_IGT['G'].to_csv('{}\{}_Y_i_data.csv'.format(root,'G'),header = False,index = False)
        dfs_IGT['R'].to_csv('{}\{}_Z_i_data.csv'.format(root,'R'),header = False,index = False)
        dfs_mean['E'].to_csv('{}\{}_X_m_data.csv'.format(root,'E'),header = False,index = False)
        dfs_mean['G'].to_csv('{}\{}_Y_m_data.csv'.format(root,'G'),header = False,index = False)
        dfs_mean['R'].to_csv('{}\{}_Z_m_data.csv'.format(root,'R'),header = False,index = False)
        
    #%% data analysis
    plt.close('all')
    
    substances = ['Copper','Methomyl','Zinc']
    fig,axe = plt.subplots(nrows = 3,ncols = 3,figsize = (20,15),sharex = True,sharey = True)
    
    for x,c in enumerate(substances):
        axe[0,x].set_title(c,fontsize = 18)
        
    for i,s in enumerate(IGT_s.species):
        axe[i,0].set_ylabel(IGT_s.species[s],fontsize = 16)
        
    #xlabel
    fig.text(0.5, 0.04, 'Time (mins)', ha='center',fontsize = 18)
    
    for i,s in enumerate(IGT_s.species):
        
        df = dfs_IGT_s[s].copy()
        df.columns = ['{}{}'.format(c,i) for c in substances for i in range(4)]
        
        for x,c in enumerate(substances):
            for r in range(4):
                axe[i,x].plot(df.index,df['{}{}'.format(c,r)],color = IGT_s.species_colors[s])