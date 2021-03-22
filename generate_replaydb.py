# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 15:25:27 2021

@author: Admin
"""
    


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dataread_terrain as d_terr

def sigmoid_coeffs(m,species):
    
    
    #n = number vivant
    n = int(16*(1-m))
    
    a = {'G':3.5,'E':3.5,'R':6.0}
    b = {'G':3.0,'E':2.5,'R':2.5}
    
    if n <= 1: 
        return np.array([0])
    elif (n > 1) & (n < 7):
        x = np.arange(2,n+2)
        return 2* (-1/(1+a[species]**(-x+b[species])) + 1) + 0.15
    else:
        x = np.arange(1,n+1)
        return 2* (-1/(1+a[species]**(-x+b[species])) + 1) + 0.15
    
def percent_IGT(data,species):
    
    """
    The problem here is the assumption that all the organisms are alive - the conversion to percentage will vary depending on the number alive...
    """
    # 0-10 %
    data[data <= 40] = data[data <= 40] /4
    
    # scale 10+ %
    data[data > 40] = (np.log10(data[data > 40] - 30))*22 + 9
    
    #moving mean
    data = np.array(pd.Series(data).rolling(10).mean().fillna(0))
    
    return data
    
def save_results(ind,IGT,m,species,filename):
    
    #make df to write values to .txt file for database
    res = pd.DataFrame(columns = ['IGT','Mortalite'],index = ind)
    res['IGT'] = IGT
    res['Mortalite'] = m
    return res

def gen_txt(res,file,species):
    
    root = os.getcwd()
    os.chdir(r'D:\VP\Viewpoint_data\Suez_res')
    
    specs = {'G':'Gammarus','E':'Erpobdella','R':'Radix'}
    spec = specs[species]
    
    os.chdir(r'D:\VP\Viewpoint_data\Suez_res')
        
    filename = file.split('.')[0] + species + '.txt'
    
    res['time'] = res.index.astype(np.int64)
    
    with open(filename, 'w', newline = '\n') as f:
        
        for i in range(res.shape[0]):
            f.write('f5683285-b5fa-4be0-be99-6e20a112fad5,sensor={} mortality={},toxicityindex={} {}\n'.format(spec,res.iloc[i]['Mortalite'],res.iloc[i]['IGT'],int(res.iloc[i]['time'])))
    
    os.chdir(root)

if __name__ == '__main__':

    colors = [
        '#42f5e0','#12aefc','#1612fc','#6a00a3',
        '#8ef743','#3c8f01','#0a4001','#fc03ca',
        '#d9d200','#d96c00','#942c00','#fc2803',
        '#e089b6','#a3a3a3','#7a7a7a','#303030'
        ]
    
    root = r'D:\VP\Viewpoint_data\Suez'
    os.chdir(root)
    files = [f for f in os.listdir()]
    
    print('The following files will be studied:')
    print(files)
    
    for file in files:
    
        dfs,dfs_mean = d_terr.read_data_terrain([file])
        
        # extract relevant dfs
        for species in 'G E R'.split():
        
            df = dfs[species]
            df_mean = dfs_mean[species]
            #mean values to be used for gradually removing individuals
            means = df.mean().sort_values()
            
            # plot individually (plot16)
            fig,axe = d_terr.plot_16(df)
            
            # treat array for deaths and no deaths!...
            data_alive,data_counters = d_terr.search_dead(np.array(df),species)
            
            if ('1402.csv' in file) & (species == 'G'):
                data_alive = np.ones_like(data_alive)
            
            # m is list of mortality percentage
            m = np.ones(len(data_alive),dtype = float) - (np.sum(data_alive,axis = 1))/16
            
            
            #values in np array form
            values = np.array(df)
            
            #remove values mort
            values[data_alive == 0] = np.nan
            
            # values[i][0] < values[i][1] < values[i][2]
            values.sort()
            
            IGT = np.zeros_like(m)
            old_IGT = np.zeros_like(m)
            for i in range(len(values)):
                coeffs = sigmoid_coeffs(m[i],species)
                IGT[i] = np.sum(values[i][:len(coeffs)]**coeffs)
                if np.isnan(values[i][0]):
                    old_IGT[i] = 0
                else:
                    old_IGT[i] = np.quantile(values[i][~np.isnan(values[i])],0.1)**2
               
            # caluclate IGT from raw -> %    
            IGT = percent_IGT(IGT, species)
            fig,axe = plt.subplots(2,1,figsize = (18,9),sharex = True)
            plt.suptitle('IGT 10% vs. percent new_IGT')
            axe[0].plot(df.index,old_IGT,color = 'green')
            axe[1].plot(df.index,IGT,color = 'green')
            axe[1].tick_params(axis='x', rotation=90)
            
            res = save_results(df.index, IGT, m, species, file)
            gen_txt(res,file,species)
            
    # merge_results()