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
    from data_merge import merge_dfs
    from dope_reg import dope_read
    
    
    #%% FUNCTIONS 
    
    def dope_params(df, Tox, date):
        
        date_range = [
            df[(df['TxM'] == Tox) & (df['Start'].dt.strftime('%d/%m/%Y') == date)]['Start'].values[0],
            df[(df['TxM'] == Tox) & (df['End'].dt.strftime('%d/%m/%Y') == date)]['End'].values[0]
            ]
        dopage = date_range[0] + pd.Timedelta(seconds = 90)
        
        # row value of experiment in dope reg
        conc = df[(dope_df['TxM'] == Tox) & (dope_df['Start'].dt.strftime('%d/%m/%Y') == date)]['Concentration'].values[0]
        sub = df[(dope_df['TxM'] == Tox) & (dope_df['Start'].dt.strftime('%d/%m/%Y') == date)]['Substance'].values[0]
        molecule = df[(dope_df['TxM'] == Tox) & (dope_df['Start'].dt.strftime('%d/%m/%Y') == date)]['Molecule'].values[0]
        
        """
        etude is the number of the week of the experiment.
        Etude1 would be my first week of experiments
        """
        etude = df[(dope_df['TxM'] == Tox) & (dope_df['Start'].dt.strftime('%d/%m/%Y') == date)].index[0]//5 + 1
        return dopage,date_range,conc,sub,molecule,etude
       
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
        
        print('The following files will be merged:')
        print(files)
        
        dfs = []
        for file in files:
            df = pd.read_csv(file,sep = '\t',encoding = 'utf-16')    #read each df in directory df
            df = df[df['datatype'] == 'Locomotion']                         #store only locomotion information
        
            #Error VPCore2
            #conc,subs = df['Conc'].iloc[0],df['Sub'].iloc[0]
        
            #sort values sn = , pn = ,location = E01-16 etcc., aname = A01-04,B01-04 etc.
            df = df.sort_values(by = ['sn','pn','location','aname'])
            df = df.reset_index(drop = True)
        
            #treat time variable - this gets the days and months the wrong way round
            df['time'] = pd.to_datetime(df['stdate'] + " " + df['sttime'], format = '%d/%m/%Y %H:%M:%S')
            
            maxrows = len(df)//48
            print('Before adjustment: total rows{}'.format(len(df)))
            df = df.iloc[:maxrows*48]
            print('After adjustment: total rows{}'.format(len(df)))
            dfs.append(df)
            
            
        df = merge_dfs(dfs)
            
        #E01 etc.
        mapping = lambda a : {'E': 'Erpobdella','G':'Gammarus','R':'Radix'}[a[0]]
        df['specie'] = df['location'].map(mapping)
        
        #moi le column 'animal' n'a que des NaNs
        good_cols = ['time','location','stdate','specie','entct','inact','inadur','inadist','smlct','smldur','smldist','larct','lardur','lardist','emptyct','emptydur']
        df = df[good_cols]
        
        #create animal column from location E01 -> 1 E12 -> 12
        df['animal'] = df['location'].str[1:].astype(int)
        
        # doesn't appear necessary - what is it?
        #df = df.drop('entct',axis = 1)
        
        #%% data manipulation
        """
        Datetime is in nanoseconds, //10e9 to get seconds
        Can zero this value
        """
        #recreate abtime as seconds since first value - For some reason creates a huge diff day to day
        df['abtime'] = df['time'].astype('int64')//1e9 #convert nano
        df['abtime'] = df['abtime'] - df['abtime'][0]
        
        #total distance inadist is only zeros?
        df['dist'] = df['inadist'] + df['smldist'] + df['lardist']
        
        
        date_dopage = dfs[-1].iloc[0]['time'].strftime('%d/%m/%Y')
        dope_df = dope_read()
        
        # dopage is exact datetime
        dopage,date_range,conc,sub,molecule,etude = dope_params(dope_df,Tox,date_dopage)
        
        # add experiment columns in
        df['dose'] = conc
        df['etude'] = study_no(etude)
        df['lot'] = sub
        df['formule'] = molecule
        
        
        for species in specie:
            
            df_sp = df[df['specie'] == specie[species]]
             
            timestamps = df_sp['time'].unique()
            animals = list(range(1,17))   
            
            df_dist = pd.DataFrame(index = timestamps)
            df_emptydur = pd.DataFrame(index = timestamps)
            max_zero = []
            
            for i in animals:
                temp_df = df_sp[df_sp['animal'] == i]
                df_dist[i] = temp_df['dist'].values
                df_emptydur[i] = temp_df['emptydur'].values
                max_zero.append((df_dist[i] != df_dist[i].shift()).cumsum().value_counts().max())
                
            max_zero_dist[Tox].update({species:max_zero})