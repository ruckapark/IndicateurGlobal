# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 15:21:12 2023

@author: George
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from datetime import timedelta,datetime
from scipy import signal
from sklearn.linear_model import LogisticRegression

#%% IMPORT personal mods
os.chdir('MODS')
from data_merge import merge_dfs
from dope_reg import dope_read_extend
import dataread as d_
os.chdir('..')

#%% Relevant directories - all performed on different PCs
roots = ['765_20211022',
         '767_20211022',
         '762_20211022',
         '763_20211022',
         '762_20211028',
         '762_20220225',
         '763_20220225',
         '764_20220310',
         '765_20220310',
         '765_20220317',
         '760_20220708',
         '761_20220708',
         '762_20220708',
         '763_20220708',
         '764_20220708',
         '765_20220708',
         '766_20220708',
         '767_20220708',
         '768_20220708',
         '769_20220708',]

roots_test = []

def TLCC(df1,df2):
    """ Time lagged cross correlation for two dataframes with multiple comlumns """
    args = np.array([])
    xreg = []
    yreg = []
    
    #5 repetitions to generate more data per time point
    for y in range(5):
        
        arg_lags = np.zeros(df1.shape[1])
        for x,col in enumerate(df1.columns):
        
            i = np.random.randint(50,500) + 500*y
            
            #what if dead organism (error?)
            while True:
                serie1 = np.array(df1[col])[i:i+30]
                if np.sum(serie1<10) > 10: #only take a series where there is significant movement
                    i += 100
                else:
                    break
            serie2 = np.array(df2[col])[i:i+30]
            if np.sum(serie2<10) > 10: 
                arg_lags[x] = np.nan
                continue
        
            grad1,grad2 = serie1[1:] - serie1[:-1],serie2[1:] - serie2[:-1]
            
            if len(serie1): 
                c = np.correlate(serie1, serie2, 'full')
                lags = signal.correlation_lags(len(serie1), len(serie2))
                lag = lags[np.argmax(c)]
                arg_lags[x] = lag
                xreg.append(i)
                yreg.append(lag)
                
                plt.figure()
                plt.plot(np.arange(0,len(serie1)),serie1,color = 'blue')
                plt.plot(np.arange(0,len(serie2)),serie2,color = 'orange')
                if abs(lag) >= 1:
                    plt.plot(np.arange(0,len(serie1))+lag,serie2,color = 'red',linestyle = '--')
                    
                plt.title('Cage {}, Time into experiment {}mins'.format(col,i*20//60))
                
            else:
                arg_lags[x] = np.nan
                continue
            
        arg_lags = arg_lags[~np.isnan(arg_lags)]
        args = np.concatenate((args,arg_lags))
    
    return np.median(arg_lags),xreg,yreg

def log_reg(x,y):
    """ O.5 limit with logistic regression """
    
    clf = LogisticRegression(random_state=0).fit(x,y)
    lim = (0.5 - clf.intercept_[0])/clf.coef_[0]
    standardiser = [max(y) - 1,min(y)]
    
    return clf,lim,standardiser

def plot_distribution(val1,val2,species = 'R',figname = None):
    xlims = {'E':1000,'G':1000,'R':200}
    
    plt.figure()
    sns.histplot(val1)
    sns.histplot(val2)
    plt.xlim(0,xlims[species])
    
    if figname: plt.savefig(r'C:\Users\George\Documents\Figures\DeepReplay\{}_{}_histogram.jpg'.format(species,figname))
    
    plt.figure()
    plt.plot(np.arange(0,1,0.01),np.array([np.quantile(val1,i/100) for i in range(100)]))
    plt.plot(np.arange(0,1,0.01),np.array([np.quantile(val2,i/100) for i in range(100)]))
    
    if figname: plt.savefig(r'C:\Users\George\Documents\Figures\DeepReplay\{}_{}_QuantilePlot.jpg'.format(species,figname))
    
def find_stepcutoff(x,y,cutoffs = [500,800,1100,1400,1700,2000]):
    
    lims = []
    for i in range(0,5):
        x1,x2 = np.copy(x),np.copy(x)
        x1 = np.where((y == -i)&(x < cutoffs[i]),x1,np.nan)
        x2 = np.where((y == -(i+1))&(x < cutoffs[i+1]),x2,np.nan)
        
        x1,x2 = x1[~np.isnan(x1)],x2[~np.isnan(x2)]
        y1,y2 = -i*np.ones(len(x1)),-(i+1)*np.ones(len(x2))
        
        if len(x1) and len(x2):
            xlog,ylog = np.concatenate((x1,x2)).reshape(-1,1),np.concatenate((y1,y2))
            lim = log_reg(xlog,ylog)[1]
        else:
            lim = [0]
        lims.append(lim[0])
    return lims

def read_starttime(root):
    """ read datetime and convert to object from txt file """
    startfile = open(r'{}\start.txt'.format(root),"r")
    starttime = startfile.read()
    startfile.close()
    return datetime.strptime(starttime,'%d/%m/%Y %H:%M:%S')

"""
def plot_logit():
    def model(x):
        return 1 / (1 + np.exp(-x))
    loss = model(xloss * clf.coef_ + clf.intercept_).ravel()
    plt.plot(xloss, loss-2, color='red', linewidth=3)
    
    plt.axvline((0.5 - clf.intercept_[0])/clf.coef_[0])
"""

#%% Code

"""
% of time higher then the other
average difference between values 
"""

if __name__ == '__main__':
    
    specie = {'E': 'Erpobdella','G':'Gammarus','R':'Radix'}
    
    x,y = {},{}
    for comp,r in enumerate(roots[:3]):
        Tox = r.split('_')[0]
        
        #navigate to correct directory
        stem = [d for d in os.listdir(r'I:\TXM{}-PC'.format(Tox)) if r.split('_')[-1] in d]
        root = r'I:\TXM{}-PC\{}'.format(Tox,stem[0])
        
        #locate original and the copy (VP replay version - 'copy')
        file_og = r'{}\{}.xls.zip'.format(root,stem[0])
        file_copy = r'{}\{}.replay.xls.zip'.format(root,stem[0])
        
        #read file
        df_og,df_copy = d_.read_merge([file_og]),d_.read_merge([file_copy])
        dfs_og,dfs_copy = d_.preproc(df_og),d_.preproc(df_copy)
        
        #register
        dope_df = dope_read_extend()
        dopage = d_.dope_params(dope_df,Tox,df_og.index[0],df_og.index[-1])[0]
        
        #read start time of original video from txt file
        starttime = read_starttime(root)
            
            
        #%% Use Gammarus for lag estimates (TLCC)
        df1,df2 = dfs_og['G'],dfs_copy['G']
        indexing = min(df1.shape[0],df2.shape[0])
        df1,df2 = df1.iloc[:indexing],df2.iloc[:indexing]
        
        tlc,xreg,yreg = TLCC(df1,df2)
        x.update({comp:xreg})
        y.update({comp:yreg})
        print("Median lag between series: ",tlc)
        if abs(tlc) >= 1: 
            print('tracking lag error!')

    xreg = pd.DataFrame(index = np.arange(max([len(x[c]) for c in [*x]])),columns = [*x])
    yreg = pd.DataFrame(index = np.arange(max([len(x[c]) for c in [*x]])),columns = [*x])
    
    #%%
    for comp in [*x]:
        xreg[comp].iloc[:len(x[comp])] = x[comp]
        yreg[comp].iloc[:len(y[comp])] = y[comp]
        
    x_values,y_values = np.array(xreg,dtype = float).flatten(),np.array(yreg,dtype = float).flatten()
    x_values,y_values = x_values[~np.isnan(x_values)],y_values[~np.isnan(y_values)]
    
    lims = np.array(find_stepcutoff(x_values, y_values))
    lims_sub = {}
        
    fig_all = plt.figure(figsize=(10, 5))
    axe_all = fig_all.add_axes([0.1,0.1,0.8,0.8])    
    for lim in lims: axe_all.axvline(lim)
    axe_all.set_ylim((-6,1))
    axe_all.set_xlim((0,2200))
    
    fig_sub,axe_sub = plt.subplots(2,5,figsize = (20,7))
    for i in range(10):
        axe_all.scatter(xreg[i],yreg[i])
        
        axe_sub[i//5,i%5].scatter(xreg[i],yreg[i])
        axe_sub[i//5,i%5].set_xlim((0,2000))
        axe_sub[i//5,i%5].set_ylim((-6,1))
        
        lims_sub.update({i:np.array(find_stepcutoff(np.array(xreg[i].dropna(),dtype = float),np.array(yreg[i].dropna(),dtype = float)))})
        
        for lim in lims_sub[i]: axe_sub[i//5,i%5].axvline(lim)
        
    #check influence of PC
    #for i in range()
    
        
    """
    #%% Start with Radix
    species = 'R'
    df1,df2 = dfs_og[species],dfs_copy[species]
    indexing = min(df1.shape[0],df2.shape[0])
    df1,df2 = df1.iloc[:indexing],df2.iloc[:indexing]
    df1_m,df2_m = d_.rolling_mean(df1,5),d_.rolling_mean(df2,5)
    
    fig,axe = plt.subplots(4,4,figsize = (12,20),sharex = True)
    for i in range(16):
        axe[i//4,i%4].plot(np.arange(df1.shape[0]),df1[i+1])
        axe[i//4,i%4].plot(np.arange(df2.shape[0]),df2[i+1])
    fig.tight_layout()
    
    fig,axe = plt.subplots(4,4,figsize = (12,20),sharex = True,sharey = True)
    for i in range(16):
        axe[i//4,i%4].plot(np.arange(df1_m.shape[0]),df1_m[i+1])
        axe[i//4,i%4].plot(np.arange(df2_m.shape[0]),df2_m[i+1])
        axe[i//4,i%4].set_ylim([0,800])
    fig.tight_layout()
    
    #plot distribution comparison
    values1,values2 = df1.values.flatten(),df2.values.flatten()
    values1,values2 = values1[values1 > 0],values2[values2 > 0]
    
    plot_distribution(values1,values2,species,figname = r)
    """