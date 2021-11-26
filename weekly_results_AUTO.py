# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 10:10:12 2021

File created for the weekly run of visualising data

@author: Admin
"""

#%% IMPORTS

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from datetime import timedelta
from data_merge import merge_dfs
from dope_reg import dope_read
#import gen_pdf as pdf
import dataread as d_

#%% FUNCTIONS - plotting functions could also be put into another module    
def main_auto(etude):
    
    # loop through all ToxMate and species
    Toxs = [i for i in range(760,770)]
    specie = {'E': 'Erpobdella','G':'Gammarus','R':'Radix'}
    
    #read dope reg
    dope_df = dope_read()
    
    for Tox in Toxs:
        
        plt.close('all')
        
        #locate directory - pass if empty
        os.chdir(r'D:\VP\Viewpoint_data\TxM{}-PC\{}'.format(Tox,d_.study_no(etude)))
        if not os.listdir(): continue
        files = [file for file in os.listdir() if os.path.isfile(file)]
        
        #read in data
        df = d_.read_merge(files)
        dfs = d_.preproc(df)
        
        #save result location        
        resultdir = r'D:\VP\LabResults\{}'.format(d_.study_no(etude))
        
        #check if result dir already exists
        if not os.path.isdir(resultdir):
            os.mkdir(resultdir)        
        
        for species in specie:
            
            #select specific df
            try:
                df = dfs[species]
                deadcol = d_.remove_dead(df,species)
                if len(deadcol) > 14: continue
                df = df.drop(columns = deadcol)
                dopage,date_range,conc,sub,molecule,etude_ = d_.dope_params(dope_df,Tox,df.index[0],df.index[-1])
            except:
                continue
            
            t_mins = 5
            df_mean = d_.rolling_mean(df,t_mins)
            
            #individual plots
            fig,axe = d_.plot_16(df_mean[(df_mean.index > (dopage - pd.Timedelta(hours = 2)))&(df_mean.index < (dopage + pd.Timedelta(hours = 2)))],mark = date_range)
            fig.savefig(r'{}\{}_{}_03indmean.JPG'.format(resultdir,Tox,species))
            
            #overall mean and IGT
            mean_dist = df_mean.mean(axis = 1)
            quantile_dist = df_mean.quantile(q = 0.05, axis = 1)**2
            
            fig,axe = d_.single_plot(mean_dist,title = 'Mean distance')
            d_.dataplot_mark_dopage(axe,date_range)
            fig.savefig(r'{}\{}_{}_04mean.JPG'.format(resultdir,Tox,species))
            
            fig,axe = d_.single_plot(mean_dist[(mean_dist.index > (dopage - pd.Timedelta(hours = 2)))&(mean_dist.index < (dopage + pd.Timedelta(hours = 6)))],title = 'Mean distance - reduit')
            d_.dataplot_mark_dopage(axe,date_range)
            fig.savefig(r'{}\{}_{}_05meanshort.JPG'.format(resultdir,Tox,species))
            
            title_ = 'ToxIndex: {}({}), {}   {}-{}'.format(sub,molecule,conc,date_range[0].strftime('%d/%m'),date_range[1].strftime('%d/%m'))
            
            fig,axe = d_.single_plot(quantile_dist[(quantile_dist.index > dopage - pd.Timedelta(hours = 18)) & (quantile_dist.index < dopage + pd.Timedelta(hours = 36))])
            d_.dataplot_mark_dopage(axe,date_range)
            fig.savefig(r'{}\{}_{}_02toxind.JPG'.format(resultdir,Tox,species))
            
            #smaller plot
            IGT = quantile_dist[(quantile_dist.index > dopage - pd.Timedelta(minutes = 60)) & (quantile_dist.index < dopage + pd.Timedelta(minutes = 120))]
            fig,axe = d_.single_plot(IGT, title = title_)
            axe.set_xlabel('Tox Ind')
            d_.dataplot_mark_dopage(axe,date_range)
            fig.savefig(r'{}\{}_{}_01toxind_reduit.JPG'.format(resultdir,Tox,species))
            
        
    return None

     

if __name__ == '__main__':
                
    print('Ready!')
    #main_auto(32)