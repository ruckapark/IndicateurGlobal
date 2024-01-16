# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 08:24:00 2024

For a given substance plot from the custom register mean and IGT for each species

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
os.chdir('MODS')
from data_merge import merge_dfs
from dope_reg import dope_read_extend
import dataread as d_
os.chdir('..')

#%%

molecules = ['Copper','Zinc','Aluminium','Cobalt','Manganese','Chloroacetic','Mercury','Chromium',
             'Nitrate','Chloride','DMSO','Methomyl','MCPA','Chlorfenvinphos','Lindane','Anthracene',
             'Benzo(a)pyrene','Acetone','Benzo(b)fluoranthene','Trichlorobenzene(123)','alphaEndosulfan',
             'betaEndosulfan','DDD(2-4)','Pentachlorophenol','Tebufenozide','PiperonylButoxide',
             'Carbaryl','Chlorfen','vernolat','Chlorpyrifos','Nonylphenol','Trifluralin','4-octylphenol',
             'alpha HCH','Tetrachloroethylene','Hydrazine','Trichloroethylene','1-2Dichloroethane','124-Trichlorobenzene',
             'Benzene','Nitric acid','Biphenyl','Aldrin','Dieldrin','Arsenic','Acide Acrylique','L1000',
             'H40','MHPC724','A736','P520','2A1','Soja','Ciprofloxacin','Verapamil','Ofloxacin','Tramadol',
             'Cypermethrine','Ibuprofen','1-chlorodecane','ButyltinTrichloride','Imidacloprid','Quinoxyfen',
             'Chlorothanolil','Isodrin','Dicofol','AzinphosMethyl','Diazinon','FentinChloride','MC-LR','Methanol']

def get_dataset(data, mean=True, spike = True):
    
    specie = {'E':'Erpobdella','G':'Gammarus','R':'Radix'}
    #write dataframe
    dataset = {s:[] for s in specie}
    for i in range(len(data)):
        for s in specie:
            if mean:
                try:
                    arr = data[i].mean_[s]
                except:
                    arr = data[i].mean[s]
            else:
                try:
                    arr = data[i].IGT_[s]
                except:
                    arr = data[i].IGT[s]
            arr = arr[arr.index > -3600]
            arr = arr[arr.index < 7200]
            
            if (s == 'R') and (mean == False) and (spike == False):
                arr = arr/10
            
            dataset[s].append(arr)
            
    return dataset
    

if __name__ == '__main__':
    
    plt.close('all')
    
    specie = {'E':'Erpobdella','G':'Gammarus','R':'Radix'}
    substance = 'Zinc'
        
    #Read register for all dopage of molecule m
    df = pd.read_csv(r'D:\VP\Viewpoint_data\REGS\Molecules\{}_custom.csv'.format(substance),index_col=None)
    
    spike_data = []
    control_data = []
    
    for i in range(df.shape[0]):
        
        ## Get spike data and check if file is from one test (common)
        if df.iloc[i]['Erpobdella'] == df.iloc[i]['Gammarus'] == df.iloc[i]['Radix']:
            common_file = True
        else:
            common_file = False
        
        #get data and append
        if common_file:
            root = df.iloc[i]['Erpobdella']
            data = TOX.csvDATA(root)
        else:
            root = {}
            for s in specie:
                root.update({s:df.iloc[i]['{}'.format(specie[s])]})
            data = TOX.csvDATA_comp(root)
        spike_data.append(data)
        
        
        ## Repeat for control data
        if df.iloc[i]['Erpobdella_TEM'] == df.iloc[i]['Gammarus_TEM'] == df.iloc[i]['Radix_TEM']:
            common_file = True
        else:
            common_file = False
            
        #get data and append
        if common_file:
            root_ = df.iloc[i]['Erpobdella_TEM']
            data_ = TOX.csvDATA(root_)
        else:
            root_ = {}
            for s in specie:
                root_.update({s:df.iloc[i]['{}_TEM'.format(specie[s])]})
            data_ = TOX.csvDATA_comp(root_)
        control_data.append(data_)
        
    #%% Isolate spike and control data per species, further smooth it and write to file if write is true
    # extension = 'IGT'
    # mean = False
    
    # spike_IGT = get_dataset(spike_data,mean = False)
    # control_IGT = get_dataset(control_data,mean = False,spike = False)
    
    extension = 'mean'
    mean = True
    
    spike_mean = get_dataset(spike_data,mean = mean)
    control_mean = get_dataset(control_data,mean = mean,spike = False)
    
    #Save each series in the array as Copper0.csv, Copper1.csv etc.
    write = True
    if write:
        output = r'D:\VP\ARTICLE2\ArticleRawData\{}'.format(substance)
        for s in specie:
            output_dir = r'{}\{}'.format(output,specie[s])
            # for i in range(len(spike_IGT[s])):
            #     d_.save_series_to_csv(spike_IGT[s][i],r'{}\{}{}_{}.csv'.format(output_dir,substance,i,extension))
            #     d_.save_series_to_csv(control_IGT[s][i],r'{}\{}{}_{}_.csv'.format(output_dir,substance,i,extension))
            
            for i in range(len(spike_mean[s])):
                d_.save_series_to_csv(spike_mean[s][i],r'{}\{}{}_{}.csv'.format(output_dir,substance,i,extension))
                d_.save_series_to_csv(control_mean[s][i],r'{}\{}{}_{}_.csv'.format(output_dir,substance,i,extension))