# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 16:04:17 2021

Meausure of léthargie in aquariums reference to pipe change

AVANT_1 = one week before change
APRES_0 = week of change
APRES_1 = week of change

@author: Admin
"""

import dataread as d_
import os
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns

"""Read file and merge"""

#chdir
root = r'D:\VP\Viewpoint_data\Lethargie'
direct = 'APRES_2'
os.chdir(root)
os.chdir(direct)

#store data file in list (in case multiple?)
tests = []

#check if dir in dir
files = os.listdir()
if os.path.isdir(files[0]):
    
    print('Mulitple tests')
    for direc in files:
        
        os.chdir(direc)
        files = os.listdir()
        tests.append(d_.read_merge(files,datechange = False))
        os.chdir('..')
else:    
    tests = [d_.read_merge(files,datechange = False)]
    
for test in tests:
    dfs = d_.preproc(test)
    
    df = dfs['G']
    df = df.drop(columns = d_.remove_dead(df,'G'))
    df_mean = d_.rolling_mean(df,300)
    fig,axe = d_.plot_16(df_mean,title = '{}_{}'.format('G',direct))
    mean_dist = df_mean.mean(axis = 1)
    fig,axe = d_.single_plot(df.mean(axis = 1),title = 'Mean {}_{}'.format('G',direct))
    
    #find hourly mean with std.
    mean_df = pd.DataFrame(df.mean(axis = 1),columns = ['Mean dist'])
    mean_df['hour'] = ((df.index - df.index[0])/np.timedelta64(1,'h')).astype(int)
    
    fig = plt.figure(figsize = (15,8))
    axe = fig.add_axes([0.1,0.1,0.8,0.8])
    sns.boxplot(x = 'hour', y = 'Mean dist',data = mean_df, color = 'skyblue')
    
    start = df.index[0]
    true_start = pd.to_datetime('{} 11:00:00'.format(start.strftime("%d/%m/%Y")),format = "%d/%m/%Y %H:%M:%S")
    time_offset = (start - true_start)/np.timedelta64(1,'h')
    
    medians = mean_df.groupby('hour').median()
    y_offset = medians[-24:].mean()[0]
    
    regress_coeff = sp.stats.linregress(medians[:48].index,medians[:48]['Mean dist'])[:2]
    regress = lambda x : x*regress_coeff[0] + regress_coeff[1]
    
    #%% fit medians to exponential decay curve
    
    def fit_exp_nonlinear(t, y, p):
        opt_parms, parm_cov = sp.optimize.curve_fit(model_func, t, y,p)
        A, K = opt_parms
        return A, K

    def model_func(t, A, K):
        return A * np.exp(K * t) + y_offset
    
    #define time in secs - would be nice to zero at 11:00 on tuesday
    x_time = np.array(medians.index)
    y_test = np.array(medians['Mean dist'])
    
    A0,K0 = 300,-0.1
    
    A,K = fit_exp_nonlinear(x_time,y_test,(A0,K0))
    fit_y = model_func(x_time,A,K)
    
    axe.plot(x_time,fit_y,color = 'r',linestyle = '--',label = 'Exponential decay fit')
    axe.axhline(y_offset,color = 'black',label = 'Lethargie')
    axe.plot(x_time,regress(x_time),color = 'r',label = 'Descente')
    axe.set_xticks(x_time[0::3])
    axe.set_xticklabels(x_time[0::3] + int(time_offset))
    axe.set_title('Temoin - {} Changement tuyau'.format(direct))
    
    intersect = (y_offset - regress_coeff[1])/regress_coeff[0]
    axe.axvline(intersect,color = 'blue',linestyle = '--',label = 'Temps Lethargie - {}'.format(round(intersect+time_offset,2)))
    axe.legend()











    # #plot means for gammares, erpo, radix
    # for species in 'E G R'.split():
    #     df = dfs[species]
    #     df = df.drop(columns = d_.remove_dead(df,species))
        
        
    #     #plot individual plots
    #     df_mean = d_.rolling_mean(df,180)
    #     fig,axe = d_.plot_16(df_mean,title = '{}_{}'.format(species,direct))
    #     mean_dist = df_mean.mean(axis = 1)
    #     fig,axe = d_.single_plot(mean_dist,title = 'Mean {}_{}'.format(species,direct))
        
    #     #think about conditions for removing individuals with the least movement?
    #     derivs = df_mean.diff().mean(axis = 1)
    #     fig,axe = d_.single_plot(derivs,title = 'Derives {}_{}'.format(species,direct))
        
    #     #define moving mean that removes most noise 
        
        
    #     #look at when the gradient of said curve hits zero