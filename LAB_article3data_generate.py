# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 10:56:13 2024

Write datasets similarly to LAB_article2data_generate.py for Article 3
Could include datasets from Alex Descamps dataset
Follow datafile by repetition number and date

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

#%% Functions
def read_custom_reg(sub,reg = r'D:\VP\Viewpoint_data\REGS\Molecules'):
    specie = {'E':'Erpobdella','G':'Gammarus','R':'Radix'}
    
    f = r'{}\{}_custom.csv'.format(reg,sub)
    reg = pd.read_csv(f,index_col = 'Repetition',header = 0)
    
    repetitions = {}
    for r in reg.index.values:
        repetitions.update({r:{s:reg[specie[s]].loc[r] for s in specie}})
        
    return repetitions

#%% Registers
substances = [
    '1-2Dic','1-chlorodecane','2A1','4-octylphenol','124Tri','A736','Acetone','Acide Acrylique',
    'alpha endo','Aluminium','Anthracene','Benzene','Benzo(a)pyrene','beta endo','Biphenyl',
    'Carbaryl','Chlorothanolil','Chlorpyrifos','Cobalt','Copper','Cypermethrin','DDD','Dicofol',
    'Dieldrin','H40','Hyrdrazine','Ibuprofen','Isodrin','L1000','Lindane','Mercury','Methomyl',
    'Nitric Acid','Pentachlorophenol','PipeonylButoxide','Quinoxyfen','Soja','Tebufenozide',
    'Tetrachloroethylene','Trichlorobenzene123','Trichloroethylene','Trifluralin','Verapamil','Zinc'
    ]

#%% Main Code
if __name__ == "__main__":

    #the same as above
    datasets = {}
    for s in substances: datasets.update({s:read_custom_reg(s)})
    
    input_directory = r'D:\VP\ARTICLE2\ArticleData'  #find data means or IGTs
    
    written = False
    cols = []
    
    if not written:
        for s in datasets:
            for r in datasets[s]:
                data = TOX.csvDATA_comp(datasets[s][r])
                data.write_data(r'D:\VP\ARTICLE2\ArticleData')
    
    for s in datasets:
        for i in range(len(datasets[s])): cols.append('{}{}'.format(s,i))
    
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
                dfs_IGT[s] = pd.DataFrame(index = IGT_s.data.index,columns = cols)
                dfs_mean[s] = pd.DataFrame(index = mean_s.data.index,columns = cols)
                #smooth
                dfs_IGT_s[s] = pd.DataFrame(index = IGT_s.data.index,columns = cols)
                dfs_mean_s[s] = pd.DataFrame(index = mean_s.data.index,columns = cols)
        
        #write data to dataframes
        for s in IGT_s.active_species:
            dfs_IGT[s][cols[i]] = IGT_s.data[s].values[:len(dfs_IGT[s].index)]
            dfs_mean[s][cols[i]] = mean_s.data[s].values[:len(dfs_IGT[s].index)]
            
            dfs_IGT_s[s][cols[i]] = IGT_s.data['{}smooth'.format(s)].values[:len(dfs_IGT_s[s].index)]
            dfs_mean_s[s][cols[i]] = mean_s.data['{}smooth'.format(s)].values[:len(dfs_mean_s[s].index)]
        
        IGT_s.plot_raw()
        mean_s.plot_raw()
        
    #write dataframes to csv files - no IGT in titles to avoid clash
    if not written:
        root = r'D:\VP\ARTICLE2\ArticleData'
        dfs_IGT['E'].to_csv('{}\{}_X_i_data.csv'.format(root,'E'),index = False)
        dfs_IGT['G'].to_csv('{}\{}_Y_i_data.csv'.format(root,'G'),index = False)
        dfs_IGT['R'].to_csv('{}\{}_Z_i_data.csv'.format(root,'R'),index = False)
        dfs_mean['E'].to_csv('{}\{}_X_m_data.csv'.format(root,'E'),index = False)
        dfs_mean['G'].to_csv('{}\{}_Y_m_data.csv'.format(root,'G'),index = False)
        dfs_mean['R'].to_csv('{}\{}_Z_m_data.csv'.format(root,'R'),index = False)
        
    #%% data analysis
    plt.close('all')
    
    fig,axe = plt.subplots(nrows = len(datasets),ncols = 3,figsize = (20,20),sharex = True,sharey = True)
    
    for x,c in enumerate(datasets):
        axe[x,0].set_ylabel(c,fontsize = 18)
        
    for x,s in enumerate(IGT_s.species):
        axe[0,x].set_title(IGT_s.species[s],fontsize = 16)
        
    #xlabel
    fig.text(0.5, 0.04, 'Time (mins)', ha='center',fontsize = 18)
    
    for i,s in enumerate(IGT_s.species):
        
        df = dfs_IGT_s[s].copy()
        df.columns = cols
        #df.columns = ['{}{}'.format(c,i) for c in substances for i in range(4)]
        
        for x,sub in enumerate(datasets):
            for r in range(len(datasets[sub])):
                axe[x,i].plot(df.index,df['{}{}'.format(sub,r)],color = IGT_s.species_colors[s])