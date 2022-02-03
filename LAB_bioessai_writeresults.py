# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 10:51:54 2022

Bioessais has multiple dopages in one experiment !

@author: George
"""

#%% IMPORTS classic mods
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% IMPORT personal mods
os.chdir('MODS')
from data_merge import merge_dfs
from dope_reg import dope_read
#import gen_pdf as pdf
import dataread as d_
os.chdir('..')

#%% FUNCTIONS - plotting functions could also be put into another module    
     

if __name__ == '__main__':
    
    #study number of bioessai
    bioessai = 1
    etude = 'Bioessai_{}'.format(bioessai)
                
    # loop through all ToxMate and species
    Toxs = [i for i in range(760,770)]
    specie = {'E': 'Erpobdella','G':'Gammarus','R':'Radix'}
    
    #read dope reg
    dope_df = dope_read('bioessai_reg')
    
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
                print('dope read of mortality error : {}'.format(species))
                continue
            
            t_mins = 5
            df_mean = d_.rolling_mean(df,t_mins)
            
            #overall mean and IGT
            mean_dist = df_mean.mean(axis = 1)
            quantile_dist = df_mean.quantile(q = 0.05, axis = 1)**2
            
            if type(dopage) != pd._libs.tslibs.timestamps.Timestamp:
                
                fig_mean,axe_mean = d_.single_plot(mean_dist,title = 'Mean distance')
                title_ = 'ToxIndex: {}({}), concs : {} - {}'.format(sub,molecule,conc.iloc[0],conc.iloc[-1])
                fig_IGT,axe_IGT = d_.single_plot(quantile_dist,title = title_)
                
                for i in range(dopage.shape[0]):
                    dope = dopage.iloc[i]
                    dates = [date_range[0].iloc[i],date_range[1].iloc[i]]
            
                    d_.dataplot_mark_dopage(axe_mean,dates)
                    d_.dataplot_mark_dopage(axe_IGT,dates)
                    
                fig_mean.savefig(r'{}\{}_{}_mean.JPG'.format(resultdir,Tox,species))
                fig_IGT.savefig(r'{}\{}_{}_toxind.JPG'.format(resultdir,Tox,species))