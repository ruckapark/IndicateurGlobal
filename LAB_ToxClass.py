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
            return data.rolling(window = self.params['t'],win_type = self.method.lower(),center = True).mean(std = self.params['std']).dropna()
            
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
        self.meandata_short = self.condense_data(mean = True)
        
            
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

    def get_meandfs(self,smoothing):
        
        #should this be mean of every column or just the mean series?
        
        dfs = {s:None for s in self.species}
        for s in dfs:
            if len(self.morts[s]) > 8:
                print('{} : Excessive mortality'.format(self.species[s]))
                continue
            
            df = self.data[s].copy()
            dfs.update({s:smoothing.moving_mean(df)})
            
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
    
    def condense_data(self,mean = False): #later should be data class with entry as data with condensed data series
        
        data_short = {}
        if mean:
            dfs = self.meandata
        else:
            dfs = self.data
        for s in dfs:
            df = dfs[s].copy()
            df = df[df.index > self.dopage - pd.Timedelta(hours = 1)]
            df = df[df.index < self.dopage + pd.Timedelta(hours = 5)]
            zero_index = np.argmin(abs((self.dopage - df.index).total_seconds())) - 1
            index = (df.index - df.index[zero_index]).total_seconds()
            df = df.set_index(np.array(index,dtype = int),drop = True)
            data_short.update({s:df})
        return data_short
    
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

class ToxPLOT:
    
    def __init__(self,data):
        self.type = type(data)
        self.data = data
        
    def plot16(self,species,with_mean = True,title = None,short = True,mark = True):
        
        if short:
            df = self.data.data_short[species]
            df_m = self.data.meandata_short[species]
        else:
            df = self.data.data[species]
            df_m = self.data.meandata[species]
        
        fig,axes = plt.subplots(4,4,sharex = True,sharey = True,figsize = (20,8))
        plt.suptitle(title)
        for i in range(16):
            col = i+1
            if col in df.columns:
                axes[i//4,i%4].plot(df.index,df[col],color = self.data.species_colors[species])
                
                if with_mean:
                    axes[i//4,i%4].plot(df_m.index,df_m[col],color = 'black')
                
                if mark:
                    self.mark_spike(axes,short)
                    
        return fig,axes   
    
    def mark_spike(self,axes,short = True):
        
        for i in range(16):
            if short: 
                axes[i//4,i%4].axvline(0,color = 'black')
            else:
                axes[i//4,i%4].axvline(self.data.dopage,color = 'black')


if __name__ == '__main__':
    
    dope_df = dope_read_extend()
    data = csvDATA(r'I:\TXM768-PC\20220317-165322',dope_df)