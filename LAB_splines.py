# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 16:52:00 2023

Class to create bspline approximated data
Inspired by letter data method on github ruckapark

@author: George
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from datetime import timedelta
from scipy import interpolate

#%% IMPORT personal mods
os.chdir('MODS')
from data_merge import merge_dfs
import dataread as d_
os.chdir('..')

class R_def_call:
    
    def __init__(self):
        self.type = 'type'
        self.rangeval = 'rangeval'
        self.nbasis = 'nbasis'
        self.params = 'params'
        self.dropind = 'dropind'
        self.quadvals = 'quadvals'
        self.values = 'values'
        self.basisvalues = 'basisvalues'
        
class ToxSplines:
    
    def __init__(self,filepath):
        
        self.title = filepath.split('\\')[-1]
        self.species = {'E':'Erpobdella','G':'Gammarus','R':'Radix'}
        self.data = self.read_input(filepath)
        self.active_species = list(self.data.columns)
        self.species_colors = {'E':'#de8e0d','G':'#057d25','R':'#1607e8'}
        
        #use hanning window to smooth signal at either end?
        
        self.ticks = np.arange(self.data.shape[0])
        self.t,self.c,self.k = ({s:None for s in self.active_species} for i in range(3))
        for s in self.active_species:
            self.get_bSpline(s)
            self.data_smooth = self.eval_spline(s)      

    def read_input(self,filepath):
        
        """ Read dataframe from filepath """
        data = pd.read_csv(filepath,index_col = 0)
        data.index = data.index / 60
        return data
    
    def plot_raw(self,short = True):
        
        fig,axes = plt.subplots(len(self.active_species),1,figsize = (15,15),sharex = True)
        fig.suptitle(self.title)
        for i,s in enumerate(self.active_species):
            axes[i].plot(self.data.index,self.data[s],color = self.species_colors[s])
            axes[i].plot(self.data.index,self.data['{}smooth'.format(s)],color = 'red',linestyle = '--')
            axes[i].set_title(self.species[self.active_species[i]])
        
    
    def get_bSpline(self,col,order = 3,k = 30):
        
        #Assume optimum knots 20 (15 minute jumps)
        y = self.data[col].values
        k_t = np.linspace(0,len(y),k,dtype = np.int32)
        
        self.t[col],self.c[col],self.k[col] = interpolate.splrep(self.ticks,y,t = k_t[1:-1])
        
    def eval_spline(self,s):
        
        self.data['{}smooth'.format(s)] = interpolate.splev(self.ticks,(self.t[s],self.c[s],self.k[s]))
        
class Rbasis():
    
    def __init__(self,ToxSpl):
        self.call = R_def_call()
        self.type = 'bspline'
        self.rangeval = self.get_range(ToxSpl)
        self.nbasis = self.get_nbasis(ToxSpl)
        self.params = self.get_params(ToxSpl)
        self.dropind = None
        self.quadvals = None
        self.values = []
        self.basisvalues = []
        self.names = ['bspl4.{}'.format(i+1) for i in range(self.nbasis)]
        
    def get_range(self,data):
        
        s = data.active_species[0]
        knots = np.unique(data.t[s])
        return [knots[0],knots[-1]]
    
    def get_nbasis(self,data):
        
        s = data.active_species[0]
        n_knots = np.unique(data.t[s]).shape[0]
        n_order = data.k[s] + 1#should this be 3 or 4?
        return n_knots + n_order - 1
    
    def get_params(self,data):
        
        s = data.active_species[0]
        knots = np.unique(data.t[s])
        return knots[1:-1]
    
    


if __name__ == "__main__":
    
    plt.close('all')
    
    input_directory = r'D:\VP\ARTICLE2\ArticleData'  #find data means or IGTs
    IGTs = [f for f in os.listdir(input_directory) if 'IGT' in f]
    means = [f for f in os.listdir(input_directory) if 'mean' in f]
    
    dfs_IGT,dfs_mean = ({s:None for s in ['E','G','R']} for i in range(2))
     
    for i in range(len(IGTs)):
        IGT,mean = IGTs[i],means[i]
    
        IGT_s = ToxSplines(r'{}\{}'.format(input_directory,IGT))
        mean_s = ToxSplines(r'{}\{}'.format(input_directory,mean))
        
        if not i: 
            for s in dfs_IGT:
                dfs_IGT[s] = pd.DataFrame(index = IGT_s.data.index,columns = np.arange(len(IGTs)))
                dfs_mean[s] = pd.DataFrame(index = mean_s.data.index,columns = np.arange(len(means)))
        
        #write data to dataframes
        for s in IGT_s.active_species:
            dfs_IGT[s][i] = IGT_s.data[s].values[:len(dfs_IGT[s].index)]
            dfs_mean[s][i] = mean_s.data[s].values[:len(dfs_IGT[s].index)]
        
        IGT_s.plot_raw()
        mean_s.plot_raw()
        
    #write dataframes to csv files
    root = r'D:\VP\ARTICLE2\ArticleData'
    dfs_IGT['E'].to_csv('{}\{}_X_i_data.csv'.format(root,'E'),header = False,index = False)
    dfs_IGT['G'].to_csv('{}\{}_Y_i_data.csv'.format(root,'G'),header = False,index = False)
    dfs_IGT['R'].to_csv('{}\{}_Z_i_data.csv'.format(root,'R'),header = False,index = False)
    dfs_mean['E'].to_csv('{}\{}_X_m_data.csv'.format(root,'E'),header = False,index = False)
    dfs_mean['G'].to_csv('{}\{}_Y_m_data.csv'.format(root,'G'),header = False,index = False)
    dfs_mean['R'].to_csv('{}\{}_Z_m_data.csv'.format(root,'R'),header = False,index = False)