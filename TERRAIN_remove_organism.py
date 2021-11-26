# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 15:56:19 2021

Function to delete all living from cages.

@author: Admin
"""


if __name__ == "__main__":
    
    """
    Make, based on consecutive zeros or similar - a function to remove deceased organisms
    """    

    import os
    import pandas as pd
    import numpy as np
    import plotly.express as px
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn import preprocessing
    from datetime import timedelta
    
    #%% IMPORT personal mods
    os.chdir('MODS')
    from data_merge import merge_dfs
    from dope_reg import dope_read
    os.chdir('..')
    
    
    #%% FUNCTIONS
       
    def study_no(etude):
        if etude <10:
            etude = 'ETUDE_00{}'.format(etude)
        elif etude < 100:
            etude = 'ETUDE_0{}'.format(etude)
        else:
            etude = 'ETUDE_{}'.format(etude)
            
        return etude
    
    #%% SETUP
    
    sns.set_style('darkgrid', {"xtick.major.size": 8, "ytick.major.size": 8})
    sns.set_context('paper')
    colors = [
        '#42f5e0','#12aefc','#1612fc','#6a00a3',
        '#8ef743','#3c8f01','#0a4001','#fc03ca',
        '#d9d200','#d96c00','#942c00','#fc2803',
        '#e089b6','#a3a3a3','#7a7a7a','#303030'
        ]
           
    Toxs = [763,764,765,766,767]
    specie = {'E': 'Erpobdella','G':'Gammarus','R':'Radix'}
    
    max_zero_dist = {}
    
    
    for Tox in Toxs:
        
        max_zero_dist.update({Tox:{}})
        
        os.chdir(r'D:\VP\Viewpoint_data\TxM{}-PC\Etude007'.format(Tox))
        
        files = [file for file in os.listdir() if os.path.isfile(file)]
        df = read_merge(files)
        dfs = preproc(df)
        
        for species in specie:
            
            df_sp = dfs[species]
             
            timestamps = df_sp['time'].unique()
            animals = list(range(1,17))   \
            
            df_dist = pd.DataFrame(index = timestamps)
            df_emptydur = pd.DataFrame(index = timestamps)
            max_zero = []
            
            for i in animals:
                temp_df = df_sp[df_sp['animal'] == i]
                df_dist[i] = temp_df['dist'].values
                df_emptydur[i] = temp_df['emptydur'].values
                max_zero.append((df_dist[i] != df_dist[i].shift()).cumsum().value_counts().max())
                
            max_zero_dist[Tox].update({species:max_zero})