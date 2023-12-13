# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 10:01:18 2023

LAB_dosages for the tested molecules

@author: George
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    plt.close('all')
    
    data_table = r'D:\VP\Viewpoint_data\REGS\DosageToxPrints.csv'
    data = pd.read_csv(data_table,delimiter='\t')
    data[['REPETITION','EXPERIMENT']] = data['LABEL'].str.split('-', expand=True)
    data = data.drop('LABEL',axis = 1)
    lims = [(0,1.2),(0,1.4),(0,1.2),(0.4,1.55),(0.9,1.15)]
    methods = [
        'Method - Fluo-atomic and ICP-AES//MS',
        'Method - MT04 HSMS',
        'Method - MT02 GCMSMS',
        'Method - MT16//73//89',
        'Method - Chromato-ionic']
    
    for i,exp in enumerate(data['EXPERIMENT'].unique()):
        
        fig = plt.figure(figsize = (13,5))
        axe = fig.add_axes([0.1,0.1,0.8,0.8])
        
        df = data[data['EXPERIMENT'] == exp]
        table = pd.DataFrame(index = np.arange(6),columns = df['COMPOUND'].unique())
        for col in table.columns:
            table[col] = np.array(df[df['COMPOUND'] == col]['RESULT'].values)
            axe.plot(table[col].iloc[1:]/table[col].iloc[1],label = col)
            
        custom_labels = ['2m', '10m', '48m', '4h', '20h']
        
        axe.set_xticks(np.arange(1,6), custom_labels)
        axe.set_ylim(lims[i])
        axe.set_title(methods[i],fontsize = 17)
        fsize = 14
        if i == 2: fsize = 9.5
        if i < 2:
            loc = 'lower left'
        else:
            loc = 'upper right'
        axe.legend(loc = loc,fontsize = fsize)
        axe.set_ylabel('% Original Concentration Found',fontsize = 15)