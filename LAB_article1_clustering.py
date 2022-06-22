# -*- coding: utf-8 -*-
"""
Created on Wed 15:29:32 2022

First attemp at fingerprints

@author: George
"""

#%% IMPORTS classic mods
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from datetime import timedelta
import ruptures as rpt

#%% IMPORT personal mods
os.chdir('MODS')
from data_merge import merge_dfs
from dope_reg import dope_read
#import gen_pdf as pdf
import dataread as d_
os.chdir('..')

specie = {'E': 'Erpobdella','G':'Gammarus','R':'Radix'}

dataset = ['763_Copper5.xls',
            '764_Copper4.xls',
            '765_Copper5.xls',
            '767_Copper7.xls',
            '767_Copper9.xls',
            '760_Methomyl3.xls',
            '761_Methomyl2.xls',
            '761_Methomyl3.xls',
            '762_Methomyl1.xls',
            '769_Methomyl1.xls'
            ]

#%% code
if __name__ == "__main__":
    
    #treat each data file, add to dictionary
    data = {}
    dopages = {}
    
    for datafile in dataset:
        if 'Copper' in datafile:
            substance = 'copper'
            dope_sub = 'copper'
        else:
            substance = 'methomyl'
            dope_sub = 'meth'
        
        root = r'D:\VP\ARTICLE1_{}\Data'.format(substance)
        dope_df = dope_read('{}_reg'.format(dope_sub))
        
        #Toxname
        Tox = int(datafile[:3])
        file = r'{}\{}'.format(root,datafile)
        
        #extract only gammarus df
        df = d_.read_merge([file])
        dfs = d_.preproc(df)
        df = dfs['G']
        deadcol = d_.remove_dead(df,'G')
        if len(deadcol) > 14: continue
        df = df.drop(columns = deadcol)
        
        #meaned df
        t_mins = 5
        df_mean = d_.rolling_mean(df,t_mins)
        
        #find dopage time
        dopage,date_range,conc,sub,molecule,etude_ = d_.dope_params(dope_df,Tox,df.index[0],df.index[-1])
        
        data.update({datafile:[df,df_mean]})
        dopages.update({datafile:[dopage,date_range,conc]})
            
    #%% analysis
    change_cost = {
        'Methomyl':20,'Copper':2000
        }
    train_data = {}
    for study in data:
        
        #data
        [df,df_mean] = data[study]
        [dopage,date_range,conc] = dopages[study]
        
        mean_dist = df_mean.mean(axis = 1)
        quantile_dist = df_mean.quantile(q = 0.05, axis = 1)**2
        
        fig,axe = d_.single_plot(mean_dist,title = 'Mean : {}'.format(conc))
        d_.dataplot_mark_dopage(axe,date_range)
        
        fig,axe = d_.single_plot(quantile_dist,title = 'IGT : {}'.format(conc))
        d_.dataplot_mark_dopage(axe,date_range)
        
        points = np.zeros((21,2))
        
        #define first point as the dopage
        points[0] = [np.argmin(abs(quantile_dist.index - dopage)),quantile_dist.iloc[np.argmin(abs(quantile_dist.index - dopage))]]
        
        #define 19 more points from highest order changepoints
        sample_length = len(quantile_dist)
        no_samples = len(quantile_dist)
        
        #apply moving mean
        quant = quantile_dist.rolling(10,center = True).mean()
        
        fig,ax = plt.subplots(nrows = 2, sharex = True)
        ax[0].plot(np.array(quant))
        ax[1].plot(np.gradient(quant))
        
        #use gradient = 0
        grad = np.gradient(quant)
        
        #local minima and maxima
        zero_crossings = np.where(np.diff(np.sign(grad)))[0]
        for x in zero_crossings:
            ax[0].axvline(x)
            ax[1].axvline(x)
            
        #take the integral of the gradient either side of the changepoint and take the difference
        ordered_standingpoints = np.zeros((len(zero_crossings),2))
        for i,x in enumerate(zero_crossings):
            if (x < 10) or (len(quant) - x < 10): continue
            #avoid consecutive changes
            ordered_standingpoints[i] = [x,abs(2*quant[x] - quant[x-10] - quant[x+10])]
            if i>0:
                if (zero_crossings[i] - zero_crossings[i-1]) < 4:
                    ordered_standingpoints[i] = [x,0]
            
        ordered_standingpoints = pd.DataFrame(ordered_standingpoints[ordered_standingpoints[:, 1].argsort()][::-1],columns = ['Position','Cost'])
        
        if 'Meth' in study:
            cost = change_cost['Methomyl']
        else:
            cost = change_cost['Copper']
        
        #only take changes with sufficient spacing either side
        changes = ordered_standingpoints[ordered_standingpoints['Cost'] > cost]
        if changes.shape[0] > 20:
            changes = np.array(changes.iloc[:20])
        else:
            changes = np.array(changes)
            changes = np.vstack((changes,np.zeros((20 - len(changes),2))))
            
        for x in changes:
            ax[0].axvline(x[0], color = 'red')
            ax[1].axvline(x[0], color = 'red')
        
        for i in range(20):
            points[i+1] = [changes[i][0],quantile_dist.iloc[int(changes[i][0])]]
            
        signature = np.array(points[points[:, 0].argsort()])
        plt.figure()
        plt.plot(signature[:,0],signature[:,1])
        
        train_data.update({study.split('.')[0]:signature})