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
import LAB_readxls as xls_

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
         '769_20220708']

roots_test = []

def TLCC(df1,df2,plot = False):
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
        
            #test of using graident rather than raw value but no good.
            #grad1,grad2 = serie1[1:] - serie1[:-1],serie2[1:] - serie2[:-1]
            
            if len(serie1): 
                c = np.correlate(serie1, serie2, 'full')
                lags = signal.correlation_lags(len(serie1), len(serie2))
                lag = lags[np.argmax(c)]
                arg_lags[x] = lag
                xreg.append(i)
                yreg.append(lag)
                
                if plot:
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

def filter_gammarus(df_r,df=None):
    
    """
    df_ is the replayed version with corrected index
    df original
    
    use Radix specific algorithm to filter data and create new df
    """
    
    thresh_gammarus = {
        'high':220,
        'mid':200,
        'low':50
        }
    
    if type(df) == pd.core.frame.DataFrame:
        for col in df_r.columns:
            replay = df_r[col]
            old = df[col]
            outliers = replay[replay > thresh_gammarus['high']]
            
            for t in outliers.index:
                ind_old = old[(old.index > t - pd.Timedelta(11,'s')) & (old.index < t + pd.Timedelta(11,'s'))].index[0]
                if old.loc[ind_old] < thresh_gammarus['mid']:
                    df_r.loc[t][col] = old.loc[ind_old]
                else:
                    df_r.loc[t][col] = 0.0
                    
    else:
        
        #high outliers
        for col in df_r.columns:
            replay = df_r[col]
            outliers = replay[replay > thresh_gammarus['high']]
            
            for t in outliers.index:
                
                #surrounding values
                ds = replay[(replay.index > t - pd.Timedelta(120,'s')) & (replay.index < t + pd.Timedelta(120,'s'))]
                
                vals = ds.values
                vals.sort()
                
                if vals.shape[0]:
                    #if half values v low, remove
                    if np.mean(vals[:vals.shape[0]//2]) < 10:
                        df_r.loc[t][col] = 0.0
                
                    #if most values much lower, replace
                    elif np.mean(vals[:3*(vals.shape[0]//4)]) < thresh_gammarus['low']:
                        df_r.loc[t][col] = np.mean(vals[:3*(vals.shape[0]//4)])
                
                else: 
                    continue
            
        
            #mid outliers
            replay = df_r[col]
            outliers_mid = replay[replay > thresh_gammarus['mid']]
            
            for t in outliers_mid.index:
                
                #surrounding values from old df +- 20 seconds
                ds = replay[(replay.index > t - pd.Timedelta(120,'s')) & (replay.index < t + pd.Timedelta(120,'s'))]
                
                vals = ds.values
                vals.sort()
                
                if vals.shape[0]:
                    #if half values v low, remove
                    if np.mean(vals[:3*(vals.shape[0]//4)]) < 1:
                        df_r.loc[t][col] = 0.0
                
                    #if most values much lower, replace
                    elif np.mean(vals[:3*(vals.shape[0]//4)]) < thresh_gammarus['low']:
                        df_r.loc[t][col] = np.mean(vals[:3*(vals.shape[0]//4)])
                
                else: 
                    continue
    
    return df_r

#%% Code

"""
% of time higher then the other
average difference between values 
"""

if __name__ == '__main__':
    
    specie = {'E': 'Erpobdella','G':'Gammarus','R':'Radix'}
    
    x,y = {},{}
    for comp,r in enumerate(roots[:1]):
        Tox = int(r.split('_')[0])
        
        #navigate to correct directory
        stem = [d for d in os.listdir(r'I:\TXM{}-PC'.format(Tox)) if r.split('_')[-1] in d]
        root = r'I:\TXM{}-PC\{}'.format(Tox,stem[0])
        
        #locate original and the copy (VP replay version - 'copy')
        file_og = r'{}\{}.xls.zip'.format(root,stem[0])
        file_copy = r'{}\{}.replay.xls.zip'.format(root,stem[0])
        
        starttime = d_.read_starttime(root)
        
        #read file
        df_og,df_copy = d_.read_merge([file_og]),d_.read_merge([file_copy])
        dfs_og,dfs_copy = d_.preproc(df_og),d_.preproc(df_copy)
        dfs_og,dfs_copy = d_.calibrate(dfs_og,Tox,starttime),d_.calibrate(dfs_copy,Tox,starttime)
        
        #register
        dope_df = dope_read_extend()
        dopage = d_.dope_params(dope_df,Tox,dfs_og[[*dfs_og][0]].index[0],dfs_og[[*dfs_og][0]].index[-1])[0]
            
        #%% Use Gammarus for lag estimates (TLCC)
        df1,df2 = dfs_og['G'],dfs_copy['G']
        indexing = min(df1.shape[0],df2.shape[0])
        df1,df2 = df1.iloc[:indexing],df2.iloc[:indexing]
        
        tlc,xreg,yreg = TLCC(df1,df2,plot = False)
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
    
    #%% scatter plots
        
    fig_all = plt.figure(figsize=(10, 5))
    axe_all = fig_all.add_axes([0.1,0.1,0.8,0.8])    
    for lim in lims: axe_all.axvline(lim)
    axe_all.set_ylim((-7,2))
    axe_all.set_xlim((0,2200))
    
    fig_sub,axe_sub = plt.subplots(2,5,figsize = (20,7))
    for i in range(len(x)):
        axe_all.scatter(xreg[i],yreg[i])
        
        axe_sub[i//5,i%5].scatter(xreg[i],yreg[i])
        axe_sub[i//5,i%5].set_xlim((0,2100))
        axe_sub[i//5,i%5].set_ylim((-6,1))
        
        lims_sub.update({i:np.array(find_stepcutoff(np.array(xreg[i].dropna(),dtype = float),np.array(yreg[i].dropna(),dtype = float)))})
        
        for lim in lims_sub[i]: axe_sub[i//5,i%5].axvline(lim)
    
    #%% Show before and after
    #timedelta start to dopage (shifted represented by underscore _)
    time_index = np.array((df1.index - df1.index[0]).total_seconds())/60
    fig,axe = plt.subplots(2,1,sharex = True,sharey = True)
    axe[0].plot(time_index,np.array(df1[14]),color = 'blue')
    axe[0].plot(time_index,np.array(df2[14]),color = 'orange')
    
    correction = 0.997
    dopage_ = dopage - starttime
    index_ = np.array((df2.index - df2.index[0]).total_seconds() * correction)
    index_ = pd.to_datetime(index_*pd.Timedelta(1,unit = 's') + df1.index[0])
    df2_ = df2.copy()
    df2_.index = index_
    
    time_index1 = np.array((df1.index - df1.index[0]).total_seconds())/60
    time_index2 = np.array((df2_.index - df2_.index[0]).total_seconds())/60
    axe[1].plot(time_index1,np.array(df1[1]),color = 'blue')
    axe[1].plot(time_index2,np.array(df2_[1]),color = 'red',linestyle = '--',alpha = 0.75)
    
    #%% Calculate IGT    
    t_mins = 5
    df_m1,df_m2 = d_.rolling_mean(df1,t_mins),d_.rolling_mean(df2_,t_mins)
    
    xls_.IGT(df_m1,dopage)
    
    
    qq1,qq2 = df_m1.quantile(q = 0.05, axis = 1)**2,df_m2.quantile(q = 0.05, axis = 1)**2
    
    plt.figure()
    plt.plot(df_m1.index,np.array(qq1))
    plt.plot(df_m2.index,np.array(qq2))
    
    plt.axvline(dopage,color = 'red')