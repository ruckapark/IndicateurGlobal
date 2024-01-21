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
        self.sup_title,self.axe_title = self.get_title()
        self.species = {'E':'Erpobdella','G':'Gammarus','R':'Radix'}
        self.data = self.read_input(filepath)
        self.active_species = list(self.data.columns)
        self.species_colors = {'E':'#de8e0d','G':'#057d25','R':'#1607e8'}
        
        #use hanning window to smooth signal at either end?
        
        self.ticks = np.arange(self.data.shape[0])
        self.t,self.c,self.k = ({s:None for s in self.active_species} for i in range(3))
        for s in self.active_species:
            self.get_bSpline(s)
            self.eval_spline(s) 
            
    def get_title(self):
        title = self.title
        if 'IGT' in title:
            measure = 'Normalised Avoidance Activity (mm/20s)'
        else:
            measure = 'Normalised Mean Locomotor Activity (mm/20s)'
        substance = title.split('_')[0]
        date = title.split('_')[-1].split('.')[0]
        Tox = int(title.split('_')[1][:3])
        Apparatus = int(Tox%759)
        return 'Substance-{}   Spike-{}   Apparatus-{}/10'.format(substance,date,Apparatus),measure

    def read_input(self,filepath):
        
        """ Read dataframe from filepath of simply input df"""
        if type(filepath) == str:
            data = pd.read_csv(filepath,index_col = 0)
        else:
            data = filepath
        data.index = data.index / 60
        return data
    
    def plot_raw(self,short = True,fname = None):
        
        sns.set_style(style = 'whitegrid')
        
        fig,axes = plt.subplots(len(self.active_species),1,figsize = (7,7),sharex = True)
        fig.suptitle(self.sup_title,fontsize = 15)
        axes[0].set_title(self.axe_title,fontsize = 14)
        for i,s in enumerate(self.active_species):
            axes[i].plot(self.data.index,self.data[s],color = self.species_colors[s])
            axes[i].plot(self.data.index,self.data['{}smooth'.format(s)],color = 'red',linestyle = '--')
            axes[i].set_ylabel(self.species[self.active_species[i]],fontsize = 14)
            
            #fit within -0.8 and 0.8 unless already higher
            lims = axes[i].get_ylim()
            axe_lims = [-0.6,0.6]
            if lims[0] < axe_lims[0]:
                axe_lims[0] = lims[0]
            if lims[1] > axe_lims[1]:
                axe_lims[1] = lims[1]
            axes[i].set_ylim(axe_lims)
            
        axes[-1].set_xlabel('Time Post Spike (Minutes)',fontsize = 15)
        
        #Save figure
        if fname:            
            fig.savefig(fname)
        
    
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
    IGT = [f for f in os.listdir(input_directory) if 'IGT' in f]
    for i in IGT:
        IGT_s = ToxSplines(r'{}\{}'.format(input_directory,i))
        IGT_s.plot_raw()