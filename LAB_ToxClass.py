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

#read dopage df
dope_df = dope_read_extend()

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
    
    def __init__(self,root,dope_df = dope_df,smoothing = 'Gaussian'):
        self.root = root
        self.Tox = self.find_tox()
        self.rootfile_stem = root + r'\\' + root.split('\\')[-1].split('-')[0] + '_'
        self.colors = ['red','purple','brown','pink','grey','olive','cyan','blue','orange','green']
        self.species_colors = {'E':'#de8e0d','G':'#057d25','R':'#1607e8'}
        self.species = {'E':'Erpobdella','G':'Gammarus','R':'Radix'}
        
        #parameters for IGT calculation
        self.method = smoothing
        self.reference_distributions = {
            'E':{'median':80,'std':20},
            'G':{'median':90,'std':80}, #large std as this is false measure
            'R':{'median':6.45,'std':3}
            }
        
        self.igt_q0,self.igt_q1,self.igt_q2 = 0.05,0.129167,0.1909
        self.low_quantiles = {'E':self.igt_q1,'G':self.igt_q1,'R':self.igt_q1}
        self.high_quantiles = {'E':1-self.igt_q1,'G':1-self.igt_q0,'R':1-self.igt_q2}
        
        #read deads
        self.morts = d_.read_dead(root)
        
        # #remove species if dead (in morts)
        # self.species = self.update_species()
        
        #get info about dopage
        self.dopage_entry = self.find_dopage_entry(dope_df)
        self.dopage = self.dopage_entry['Start']
        self.date = str(self.dopage)[:10]
        
        #get dfs from csv files
        self.data = self.get_dfs()
        self.meandata = self.get_meandfs(smoothing_PARAMETERS(self.method))
        
        #get condensed data
        self.data_short = self.condense_data()
        self.meandata_short = self.condense_data(mean = True)
        
        """
        IF IT IS DESIRED TO USE RAWDATA AND FILTER QUANTILES AFTERWARDS
        #get mean from meandata
        self.mean = self.get_mean_raw(raw = False)
        self.mean_short = self.get_mean_raw(raw = False,short = True)
        
        #get quantiles from meandata
        self.q_low = self.get_quantile(raw = False,)
        self.q_low_short = self.get_quantile(raw = False,short = True)
        
        self.q_high = self.get_quantile_raw(raw = False,high = True)
        self.q_high_short = self.get_quantile_raw(raw = False,high = True,short = True)
        
        #combine to get I
        
        """
        
        #get unfiltered mean series
        self.mean_raw = self.get_mean_raw()
        self.mean_raw_short = self.get_mean_raw(short = True)
        
        #get unfiltered IGT components
        #self.q_raw_low = self.get_quantile_raw()
        #self.q_raw_low_short = self.get_quantile_raw(short = True)
        
        #self.q_low = self.get_quantile()
        #self.q_low_short = self.get_quantile(short = True)
        
        self.q_low = self.get_quantile_raw(raw = False)
        self.q_low_short = self.get_quantile_raw(raw = False,short = True)
        
        self.q_raw_high = self.get_quantile_raw(high = True)
        self.q_raw_high_short = self.get_quantile_raw(high = True,short = True)
        
        #filter series' with moving averages
        self.mean = self.get_mean()
        self.mean_short = self.get_mean(short = True)
        
        self.q_high = self.get_quantile(high = True)
        self.q_high_short = self.get_quantile(high = True,short = True)
        
        #check if datasets are representative to calculate IGT
        self.parameters = self.check_reference()
        self.active_species = self.check_activespecies()
        
        #lower high quantile
        self.q_high_adj = self.adjust_qhigh()
        self.q_high_adj_short = self.adjust_qhigh(short = True)
        
        #calculate IGT
        self.IGT = self.combine_IGT()
        self.IGT_short = self.combine_IGT(short = True)
        
            
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
            
            if len(d_.check_dead(df,s)): 
                print('Check mortality for: ',self.species[s],d_.check_dead(df,s))
            
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
            df = df[df.index > self.dopage - pd.Timedelta(hours = 1.1)] #some margin
            df = df[df.index < self.dopage + pd.Timedelta(hours = 6.1)] #gives some margin
            zero_index = np.argmin(abs((self.dopage - df.index).total_seconds())) - 1
            index = (df.index - df.index[zero_index]).total_seconds()
            df = df.set_index(np.array(index,dtype = int),drop = True)
            data_short.update({s:df})
        return data_short
    
    def check_reference(self):
        
        """
        Check if the quantiles are in line with reference distributions
        If so, return the parameters of the spike distribution
        """
        refs = {s:None for s in self.species}
        
        for s in self.species:
            
            q_pre = self.q_high_short[s].copy()
            q_pre = q_pre[q_pre.index < 0]
            q_params = {'median':q_pre.median(),'std':q_pre.std()}
            
            lower_bound = self.reference_distributions[s]['median'] - self.reference_distributions[s]['std']
            upper_bound = self.reference_distributions[s]['median'] + self.reference_distributions[s]['std']
            
            if lower_bound < q_params['median'] < upper_bound:
                refs[s] = q_params
                
        return refs
    
    def check_activespecies(self):
        """ Return list of active species """
        return [s for s in self.species if self.parameters[s]]
    
    
    def get_mean_raw(self,raw = True,short = False):
        """ 
        return mean timeseires of df for all species
        NO FILTERING
        """
        if raw:
            if short:
                return {s:self.data_short[s].mean(axis = 1) for s in self.species}
            else:
                return {s:self.data[s].mean(axis = 1) for s in self.species}
        else:
            if short:
                return {s:self.meandata_short[s].mean(axis = 1) for s in self.species}
            else:
                return {s:self.meandata[s].mean(axis = 1) for s in self.species}
    
        
    def get_quantile_raw(self,raw = True,short = False,high = False):
        """ 
        return quantile timeseires of df for all species
        NO FILTERING
        """
        if high:
            quantiles = self.high_quantiles
        else:
            quantiles = self.low_quantiles
        
        if raw:        
            if short:
                return {s:self.data_short[s].quantile(quantiles[s],axis = 1) for s in self.species}
            else:
                return {s:self.data[s].quantile(quantiles[s],axis = 1) for s in self.species}
        else:
            if short:
                return {s:self.meandata_short[s].quantile(quantiles[s],axis = 1) for s in self.species}
            else:
                return {s:self.meandata[s].quantile(quantiles[s],axis = 1) for s in self.species}
    
    
    def get_mean(self, short=False):
        """ Overall mean of all data """
        mean = {s:None for s in self.species}
        for s in mean:
            if short:
                mean[s] = smoothing_PARAMETERS(self.method).moving_mean(self.mean_raw_short[s])
            else:
                mean[s] = smoothing_PARAMETERS(self.method).moving_mean(self.mean_raw[s])
        return mean
    
    
    def get_quantile(self, short = False, high = False, smoothing='Gaussian'):
        """ Overall IGT data """
        q = {s:None for s in self.species}
        for s in q:
            if short:
                if high:
                    q[s] = smoothing_PARAMETERS(self.method).moving_mean(self.q_raw_high_short[s])
                else:
                    q[s] = smoothing_PARAMETERS(self.method).moving_mean(self.q_raw_low_short[s])
            else:
                if high:
                    q[s] = smoothing_PARAMETERS(self.method).moving_mean(self.q_raw_high[s])
                else:
                    q[s] = smoothing_PARAMETERS(self.method).moving_mean(self.q_raw_low[s])  
        return q
    
    
    def adjust_qhigh(self,short = False):
        
        q_high_adj = {s:None for s in self.species}
        
        for s in [s for s in self.species if s in self.active_species]:
            if short:
                quantile_high = self.q_high_short[s] - (self.parameters[s]['median']-2*self.parameters[s]['std'])
            else:
                quantile_high = self.q_high[s] - (self.parameters[s]['median']-2*self.parameters[s]['std'])
            
            quantile_high[quantile_high >= 0] = 0.0
            q_high_adj[s] = quantile_high
        
        return q_high_adj
    
    
    def combine_IGT(self,short = False):
        """ Combine high and low IGT to one signal """
        
        IGTs = {s:None for s in self.species}
        
        for s in [s for s in self.species if s in self.active_species]:
            
            if short:
                qlow,qhigh = self.q_low_short[s].copy(),self.q_high_adj_short[s].copy()
            else:
                qlow,qhigh = self.q_low[s].copy(),self.q_high_adj[s].copy()
                
                
            qlow = qlow**2
            qhigh = -(qhigh**2)
            
            qlow[qlow < 0.2] = 0.0
            qhigh[qhigh > -0.2] = 0.0
            
            IGT = pd.DataFrame(index = qlow.index,columns = ['low','high','total'])
            IGT['low'],IGT['high'] = qlow,qhigh
            IGT['total'] = (1*(IGT['low'] > 0)) * (1*(IGT['high'] < 0))
            
            IGT_array = np.zeros(IGT.shape[0])
            
            for i in range(IGT.shape[0]):
                
                if IGT['total'].iloc[i] == 0:
                    if IGT['low'].iloc[i] > 0:
                        IGT_array[i] = IGT['low'].iloc[i]
                    else:
                        IGT_array[i] = IGT['high'].iloc[i]
                else:
                    if IGT['low'].iloc[i] > abs(IGT['high'].iloc[i]):
                        IGT_array[i] = IGT['low'].iloc[i]
                    else:
                        IGT_array[i] = IGT['high'].iloc[i]
            
            IGTs[s] = pd.Series(IGT_array,index = IGT.index)
        return IGTs
    
    def write_data(self,directory,short = True):
        
        if short:
            IGT,mean = pd.DataFrame(self.IGT_short),pd.DataFrame(self.mean_short)
        else:
            IGT,mean = pd.DataFrame(self.IGT),pd.DataFrame(self.mean)
            
        f_IGT = '{}_{}IGT_{}'.format(self.dopage_entry['Substance'],self.Tox,self.date)
        f_mean = '{}_{}mean_{}'.format(self.dopage_entry['Substance'],self.Tox,self.date)
        
        if short:
            IGT[IGT.index >= 0].to_csv(r'{}\{}.csv'.format(directory,f_IGT))
            mean[mean.index >= 0].to_csv(r'{}\{}.csv'.format(directory,f_mean))
        else:
            IGT.to_csv(r'{}\{}.csv'.format(directory,f_IGT))
            mean.to_csv(r'{}\{}.csv'.format(directory,f_mean))

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
                
    def plotIGT(self,spec = None,short = True):
        
        if spec:
            fig = plt.figure(figsize = (13,8))
            axe = fig.add_axes([0.1,0.1,0.8,0.8])
            axe.plot(series.index,series)
            axe.set_title(self.data.species[spec])
        else:
            
            fig,axes = plt.subplots(3,1,figsize = (15,15),sharex = True)
            for i,s in enumerate(data.species):
                if short:
                    axes[i].plot(data.IGT_short[s].index,data.IGT_short[s].values,color = data.species_colors[s])
                else:
                    axes[i].plot(data.IGT[s].index,data.IGT[s].values,color = data.species_colors[s])
                axes[i].set_title(self.data.species[s])

if __name__ == '__main__':
    
    dope_df = dope_read_extend()
    data = csvDATA(r'I:\TXM760-PC\20210625-093621',dope_df)
    ToxPLOT(data).plotIGT() #gammarus IGT needs verifying!
    #data.write_data(r'D:\VP\ARTICLE2\ArticleData')