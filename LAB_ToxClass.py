# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 16:02:20 2023

B Spline approxiamtion of IGT curve

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
os.chdir('MODS')
from data_merge import merge_dfs
from dope_reg import dope_read_extend
import dataread as d_
os.chdir('..')

class smoothing_PARAMETERS:
    
    def __init__(self,smoothing):
        self.method = smoothing
        self.params = self.get_parameters()
        
    def get_parameters(self):
        
        #all of these could of course be individual classes
        if self.method == 'Mean':
            timestep = 5
            t = timestep * 3
            return {'t':t}
        
        elif self.method == 'Gaussian':
            timestep = 5
            t = timestep * 3
            alpha = t/8
            stdev = (t-1)/(2*alpha)
            return {'t':t,'std':stdev}
        
        elif self.method == 'Exponential':
            timestep = 5
            t = timestep * 3
            tau = 3
            return {'t':t,'tau':tau}
        
        elif self.method == 'Exponential single':
            a = 0.15
            return {'alpha':a}
        
        elif self.method == 'Exponential double':
            a,b = 0.15,0.07
            return {'alpha':a,'beta':b}
        
        else:
            print('Smoothing method unknown')
            return None
            
    def moving_mean(self,data):
        """ data is series (pd or np) with index """
        
        if self.method == 'Mean':
            print('Smoothing method unknown')
            return None
        
        elif self.method == 'Gaussian':
            return data.rolling(window = self.parameters['t'],win_type = self.method.lower(),center = True).mean(std = self.params['std']).dropna()
            
        elif self.method == 'Exponential':
            print('Smoothing method unknown')
            return None
        
        elif self.method == 'Exponential single':
            print('Smoothing method unknown')
            return None
            
        elif self.method == 'Exponential double':
            print('Smoothing method unknown')
            return None
        
        else:
            print('Smoothing method unknown')
            return None

class csvDATA:
    
    def __init__(self,root,dope_df,smoothing = 'Gaussian'):
        self.root = root
        self.Tox = self.find_tox()
        self.rootfile_stem = root + r'\\' + root.split('\\')[-1].split('-')[0] + '_'
        self.colors = ['red','purple','brown','pink','grey','olive','cyan','blue','orange','green']
        self.species_colors = {'E':'#de8e0d','G':'#057d25','R':'#1607e8'}
        self.species = {'E':'Erpobdella','G':'Gammarus','R':'Radix'}
        self.morts = d_.read_dead(root)
        self.data = self.get_dfs()
        self.meandata = self.get_meandfs(smoothing_PARAMETERS(smoothing))
        self.dopage_entry = self.find_dopage_entry(dope_df)
        self.dopage = self.dopage_entry['Start']
        self.date = str(self.dopage)[:10]
        self.data_short = self.condense_data()
        
            
    def find_rootstem(self):
        return self.root + r'\\' + self.root.split('\\')[-1].split('-')[0] + '_'
    
    def find_tox(self):
        return int(self.root.split('\\')[1].split('-')[0][-3:])
    
    def get_dfs(self):
        
        dfs = {s:None for s in self.species}
        for s in dfs:
            if len(self.morts[s]) > 8:
                print('{} : Excessive mortality'.format(self.species[s]))
                continue
            
            df = pd.read_csv(r'{}{}.csv.zip'.format(self.rootfile_stem,self.species[s]),index_col = 0)
            df = self.preproc_csv(df)
            
            if len(d_.check_dead(df,s)): print('Check mortality for: ',self.species[s],d_.check_dead(df,s))
            
            for m in self.morts[s]:
                if m in df.columns: 
                    df = df.drop(columns = [m])
                    
            dfs.update({s:df})
        return dfs            
            
    def preproc_csv(self,df):
        df.index = pd.to_datetime(df.index)
        df.columns = [int(c) for c in df.columns]
        return df             
        
    
    def find_dopage_entry(self,dope_df):
        row = None
        for i in range(dope_df.shape[0]):
            if self.root.split('\\')[-1] in dope_df.iloc[i]['root']: 
                row = i
                break
        if row:
            return dope_df.iloc[row]
        else:
            print('No dopage found')
            return row
    
    def condense_data(self): #later should be data class with entry as data with condensed data series
        
        data_short = {}
        for s in self.data:
            df = self.data[s].copy()
            df = df[df.index > self.dopage - pd.Timedelta(hours = 1)]
            df = df[df.index < self.dopage + pd.Timedelta(hours = 5)]
            zero_index = np.argmin(abs((self.dopage - df.index).total_seconds())) - 1
            index = (df.index - df.index[zero_index]).total_seconds()
            df = df.set_index(np.array(index,dtype = int),drop = True)
            data_short.update({s:df})
        return data_short
    
    def get_meandfs(self,smoothing):
        
        return None
    
    def bSpline(self,i,col,order = 3,k = 10):
        """ Assume optimum knots 10 """
        x,y = self.x[col],self.y[col]
        x = x[~np.isnan(x)]
        y = y[~np.isnan(y)]
        ticks = np.arange(0,len(x))
        
        k_t = np.linspace(0,len(x),k,dtype = np.int32)
        
        tx,cx,kx = interpolate.splrep(ticks,x,t = k_t[1:-1])
        ty,cy,ky = interpolate.splrep(ticks,y,t = k_t[1:-1])
        
        #total_abs_error_x = np.sum(np.abs(x - interpolate.splev(ticks,(tx,cx,kx))))
        #total_abs_error_y = np.sum(np.abs(x - interpolate.splev(ticks,(ty,cy,ky))))
        
        self.coefficients.iloc[i,:] = [tx,cx,kx,ty,cy,ky] 
        
        #make all smoothing cooefficients equal
        self.x_smooth[col][:len(x)] = interpolate.splev(ticks,(tx,cx,kx))
        self.y_smooth[col][:len(y)] = interpolate.splev(ticks,(ty,cy,ky))

"""        
class speciesDATA():
    
    #child class from csv Data, one for each species
    return None
"""



if __name__ == '__main__':
    
    dope_df = dope_read_extend()
    data = csvDATA(r'I:\TXM768-PC\20220317-165322',dope_df)