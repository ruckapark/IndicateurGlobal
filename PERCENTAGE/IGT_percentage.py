# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 09:10:55 2021

Example passer en pourcentage

@author: Admin
"""


# modules

import os
import pandas as pd
import numpy as np
import shutil
import matplotlib.pyplot as plt

# functions


def sigmoid_coeffs(m,species):
    
    """
    Return sigmoid function coefficients,
    f ( # morts )
    if m = 0 -> return 16 coefficients
    if m = 0.5 -> return 8 coefficients
    if m = 0.0625 -> return 1 coefficient
    
    
    SIGMOID FUNC : 2 * -[ ( 1 / ( 1 + a**(x + b) ) ) + 1 ]
    """
    
    #n = # organismes vivants
    n = int(16*(1-m))
    
    # function coefficients for 3 especes
    a = {'G':3.5,'E':3.5,'R':6.0}
    b = {'G':3.0,'E':2.5,'R':2.5}
    
    # if 0,1 vivant, else if 2-7 vivant, else 7-16 vivant
    # if mortality > 0.92, elif 0.6 < mortality < 0.92, else
    if n <= 1: 
        return np.array([0])
    elif (n > 1) & (n < 7):
        x = np.arange(2,n+2)
        return 2* (-1/(1+a[species]**(-x+b[species])) + 1) + 0.15
    else:
        x = np.arange(1,n+1)
        return 2* (-1/(1+a[species]**(-x+b[species])) + 1) + 0.15


def sigmoid(values,coeffs):
    
    """In place of IGT, use sigmoid coefficients"""
    
    IGT = 0
    for i,coeff in enumerate(coeffs):
        IGT += values[i] ** coeff
        
    return IGT
    
def percent_IGT(data,species):
    
    """
    scale history of IGT from sigmoid RAW -> %
    """
    # 0-10 %
    data[data <= 40] = data[data <= 40] /4
    # scale 10+ %
    data[data > 40] = (np.log10(data[data > 40] - 30))*22 + 9
    
    #moving mean
    data = np.array(pd.Series(data).rolling(10).mean().fillna(0))
    
    return data

def IGT_per(df,alive,m,species):
    """
    Combine functions simgoid and percentage to one function
    """
    data = np.array(df)
    data[alive == 0] ==  np.nan
    data.sort()
    output = np.zeros_like(m)
    for i in range(len(output)):
        coeffs = sigmoid_coeffs(m[i],species)
        output[i] = np.sum(data[i][:len(coeffs)]**coeffs)
    return percent_IGT(output,species)
        
        
        
        
if __name__ == '__main__':
    
    file = r'toxmate_2703_2903.csv'
    
    # import general functions
    root = os.getcwd()
    os.chdir('..')
    import dataread_terrain as d_
    os.chdir(root)
    
    #return three dataframes (un par espece)
    #read data includes data wrangling
    dfs,dfs_mean = d_.read_data_terrain([file])
    
    for species in 'G E R'.split():
        
        #select df
        df = dfs[species]
        print(df.head())
        df_mean = dfs_mean[species]
        
        # plot 16 curves for dataset
        fig,axe = d_.plot_16(df_mean)
        
        #find deaths (in full history)
        #1 = alive, 0 = dead
        data_alive,data_counters = d_.search_dead(np.array(df),species)
        
        #mortality percentage 0 = no dead, 1 = all dead
        m = np.ones(len(data_alive),dtype = float) - (np.sum(data_alive,axis = 1))/16
        
        #numpy array
        values = np.array(df)
        
        #np.nan for dead value
        values[data_alive == 0] = np.nan
        
        # values[i][0] < values[i][1] < values[i][2]
        #sort axis = 1 default
        values.sort()
        
        #compare new and old IGT
        IGT = np.zeros_like(m)
        old_IGT = np.zeros_like(m)
        
        # loop through time bins
        for timebin in range(len(values)):
            
            #calculate all necessary coefficients
            coeffs = d_.sigmoid_coeffs(m[timebin],species)
            
            #calculate newIGT raw per timebin
            IGT[timebin] = np.sum(values[timebin][:len(coeffs)]**coeffs)
            """ Alternative : IGT[timebin] = sigmoid(values[timebin],coeffs)"""
            
            #check if all values nan (100% mortality)
            if np.isnan(values[timebin][0]):
                old_IGT[timebin] = 0
            else:
                old_IGT[timebin] = np.quantile(values[timebin][~np.isnan(values[timebin])],0.05)**2
           
        # caluclate IGT from raw -> %
        IGT = d_.percent_IGT(IGT, species)
        
        #compare old and new values
        fig,axe = plt.subplots(2,1,figsize = (18,9),sharex = True)
        plt.suptitle('IGT 10% vs. percent new_IGT')
        axe[0].plot(df.index,old_IGT,color = 'green')
        axe[1].plot(df.index,IGT,color = 'green')
        axe[1].tick_params(axis='x', rotation=90)
