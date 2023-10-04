# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 14:59:18 2023

Use Article 2 dataset to improve data quality

@author: George
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from datetime import timedelta
import LAB_readcsv as LABcsv

os.chdir('MODS')
from data_merge import merge_dfs
from dope_reg import dope_read_extend
import dataread as d_
os.chdir('..')

#entry 1 ok; entry 2 ok; entry 3 bad; entry 4 unsure; entry 5 bad Gammarus
#morts treated
verapamils = [
    r'I:\TXM762-PC\20220225-090938',
    r'I:\TXM763-PC\20220225-090838',
    r'I:\TXM764-PC\20220310-113652',
    r'I:\TXM765-PC\20220310-113707',
    r'I:\TXM765-PC\20220317-164730']

# 1 no radix data,2, 3 quite odd, 4 good
# morts treated
tramadols = [
    r'I:\TXM767-PC\20220225-091008',
    r'I:\TXM768-PC\20220225-090953',
    r'I:\TXM769-PC\20220310-113807',
    r'I:\TXM769-PC\20220317-164759']

#morts treated
coppers = [
    r'I:\TXM765-PC\20210422-111620',
    r'I:\TXM767-PC\20210430-124553',
    r'I:\TXM767-PC\20210513-231929',
    r'I:\TXM763-PC\20210528-113951',
    r'I:\TXM765-PC\20210326-154629',
    r'I:\TXM765-PC\20210402-160649',
    r'I:\TXM763-PC\20210402-160442',
    r'I:\TXM764-PC\20210409-135918'
    ]

#morts treated
zincs = [
    r'I:\TXM763-PC\20210416-113757',
    r'I:\TXM763-PC\20210506-230746',
    r'I:\TXM763-PC\20210513-230658',
    r'I:\TXM763-PC\20210520-224858',
    r'I:\TXM763-PC\20210430-125225',
    r'I:\TXM763-PC\20210325-104933',
    r'I:\TXM763-PC\20210409-140200',
    r'I:\TXM763-PC\20210422-111813',
    
    ]

methomyls = [
    r'I:\TXM760-PC\20210520-224501',
    r'I:\TXM760-PC\20210625-093621',
    r'I:\TXM761-PC\20210520-224549',
    r'I:\TXM761-PC\20210625-093641'
    ]

dics = [
    r'I:\TXM765-PC\20211022-095148',
    r'I:\TXM767-PC\20211022-100308',
    r'I:\TXM765-PC\20211029-075240'      
    ]

#760 data in doubt
hydrazines = [
    r'I:\TXM760-PC\20211022-095044',
    r'I:\TXM761-PC\20211022-102804',
    r'I:\TXM760-PC\20211029-074210',
    r'I:\TXM761-PC\20211029-074545'
    ]

#760 data in doubt
#761 data in doubt
ciprofloxacins = [
    r'I:\TXM760-PC\20220225-090811',
    #r'I:\TXM761-PC\20220225-091850',
    r'I:\TXM762-PC\20220310-113536',
    ]

ofloxacins = [
    r'I:\TXM764-PC\20220225-091550',
    r'I:\TXM765-PC\20220225-090923',
    r'I:\TXM767-PC\20220310-113737',
    r'I:\TXM768-PC\20220310-113753',
    r'I:\TXM767-PC\20220317-164814',
    r'I:\TXM768-PC\20220317-165322'
    ]


studies = {
    #'Copper':coppers,
    #'Methomyl':methomyls,
    #'Tramadol':tramadols,
    #'Verapamil':verapamils,
    #'Zinc':zincs,
    'Hydrazine':hydrazines,
    '12Dichloroethane':dics,
    'Ciprofloxacin':ciprofloxacins,
    'Ofloxacin':ofloxacins}

ylims = {'E':150,'G':150,'R':15}

#%% Main code
if __name__ == '__main__':
    
    for substance in [*studies]:
        for study in studies[substance]:
            plt.close('all')
            dfs = LABcsv.main(study)
            figure_dir = r'C:\Users\George\Documents\Figures\Article2\{}'.format(substance)
            
            #%% dopage
            dope_df = dope_read_extend()
            dopage_entry = LABcsv.find_dopage_entry(dope_df, study)
            dopage = dopage_entry['Start']
            dope_range = [dopage_entry['Start'],dopage_entry['End']]
            
            #%%
            for s in [*dfs]:
                df = dfs[s]
                
                #mean treatment of data
                t_mins = 5
                df_mean = d_.rolling_mean(df,t_mins)
                d_.plot_16(df_mean,mark = dope_range)
                
                #IGT
                # quantile_distRAW = df.quantile(q = 0.10, axis = 1)**2
                # quantile_dist = df_mean.quantile(q = 0.10, axis = 1)**2
                
                # fig,axe = d_.single_plot(quantile_dist)
                # axe.axvline(dopage,color = 'red')
                
                low_quantiles,high_quantiles = [0.05,0.1,0.15,0.25],[0.95,0.9,0.85,0.75]
                styles = ['solid','dashed','dashdot','solid']
                
                #full lower quantile plot
                fig = plt.figure(figsize = (11,7))
                with sns.axes_style("white"):
                    axe = fig.add_axes([0.1,0.1,0.8,0.8])
                    
                    for i in range(4):
                        axe.plot(df_mean.quantile(low_quantiles[i],axis = 1),'#a61919',alpha = 1 - 0.1*i,linestyle = styles[i],label = 'Quantile {}'.format(low_quantiles[i]))
                        
                    axe.axvline(dopage,color = 'black',label = 'Dopage')
                    
                    axe.set_xlabel('Time $(hours)$', fontsize = 20)
                    axe.set_ylabel('Distance $(mm\cdot20s^{-1})$', fontsize = 20)
                    
                    axe.set_title(study)
                    axe.legend(fontsize = 17, loc = 'upper right')
                    
                    plt.tight_layout()
                    
                fig.savefig(r'{}\{}_LowQuantile_Study{}'.format(figure_dir,s,study.split('\\')[-1]))
                
                #full higher quantile plot
                fig = plt.figure(figsize = (11,7))
                with sns.axes_style("white"):
                    axe = fig.add_axes([0.1,0.1,0.8,0.8])
                    
                    for i in range(4):
                        axe.plot(df_mean.quantile(high_quantiles[i],axis = 1),'#387d02',alpha = 1 - 0.1*i,linestyle = styles[i],label = 'Quantile {}'.format(high_quantiles[i]))
                    
                    axe.axvline(dopage,color = 'black',label = 'Dopage')
                    
                    axe.set_xlabel('Time $(hours)$', fontsize = 20)
                    axe.set_ylabel('Distance $(mm\cdot20s^{-1})$', fontsize = 20)
                    
                    axe.set_title(study)
                    axe.legend(fontsize = 17, loc = 'upper right')
                    
                    axe.set_ylim((0,ylims[s]))
                    
                    plt.tight_layout()
                    
                fig.savefig(r'{}\{}_HighQuantile_Study{}'.format(figure_dir,s,study.split('\\')[-1]))
                
                #only do for one hour before and 6 hours afterwards
                df_mean = df_mean[(df_mean.index > dopage - pd.Timedelta(hours = 1)) & (df_mean.index < dopage + pd.Timedelta(hours = 6))]
                fig = plt.figure(figsize = (11,7))
                with sns.axes_style("white"):
                    axe = fig.add_axes([0.1,0.1,0.8,0.8])
                    
                    axe.fill_between(df_mean.index,df_mean.quantile(0.25,axis = 1),df_mean.quantile(0.75,axis = 1),color = '#1492c4',alpha = 0.75,zorder = 2,label = 'Interquartile range')
                    axe.plot(df_mean.median(axis = 1),color = '#2d04c2',label = 'Median')
                    
                    for i in range(3):
                        axe.plot(df_mean.quantile(low_quantiles[i],axis = 1),'#a61919',alpha = 1 - 0.1*i,linestyle = styles[i],label = 'Quantile {}'.format(low_quantiles[i]))
                        axe.plot(df_mean.quantile(high_quantiles[i],axis = 1),'#387d02',alpha = 1 - 0.1*i,linestyle = styles[i],label = 'Quantile {}'.format(high_quantiles[i]))
                    
                    axe.axvline(dopage,color = 'black',label = 'Dopage')
                    
                    axe.set_xlabel('Time $(hours)$', fontsize = 20)
                    axe.set_ylabel('Distance $(mm\cdot20s^{-1})$', fontsize = 20)
                    
                    axe.set_title(study)
                    axe.legend(fontsize = 17, loc = 'upper right')
                    
                    axe.set_ylim((0,ylims[s]))
                    
                    plt.tight_layout()
                    
                    fig.savefig(r'{}\{}_Full_Study{}'.format(figure_dir,s,study.split('\\')[-1]))