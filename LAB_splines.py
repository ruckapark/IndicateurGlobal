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

if __name__ == "__main__":
    
    input_directory = r'D:\VP\ARTICLE2\ArticleData'  #find data means or IGTs
    IGTs = [f for f in os.listdir(input_directory) if 'IGT' in f]
    means = [f for f in os.listdir(input_directory) if 'mean' in f]
    
    for i in range(len(IGTs)):
        IGT,mean = IGTs[i],means[i]
    
        IGT_s = ToxSplines(r'{}\{}'.format(input_directory,IGT))
        mean_s = ToxSplines(r'{}\{}'.format(input_directory,mean))
        
        IGT_s.plot_raw()
        mean_s.plot_raw()