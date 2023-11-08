# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 17:49:48 2023

Find outliers MANUALLY IN THE DATA

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
import LAB_ToxClass as TOX

molecules = ['Copper','Zinc','Aluminium','Cobalt','Manganese','Chloroacetic','Mercury','Chromium',
             'Nitrate','Chloride','DMSO','Methomyl','MCPA','Chlorfenvinphos','Lindane','Anthracene',
             'Benzo(a)pyrene','Acetone','Benzo(b)fluoranthene','Trichlorobenzene(123)','alphaEndosulfan',
             'betaEndosulfan','DDD(2-4)','Pentachlorophenol','Tebufenozide','PiperonylButoxide',
             'Carbaryl','Chlorfen','vernolat','Chlorpyrifos','Nonylphenol','Trifluralin','4-octylphenol',
             'alpha HCH','Tetrachloroethylene','Hydrazine','Trichloroethylene','1-2Dichloroethane','124-Trichlorobenzene',
             'Benzene','Nitric acid','Biphenyl','Aldrin','Dieldrin','Arsenic','Acide Acrylique','L1000',
             'H40','MHPC724','A736','P520','2A1','Soja','Ciprofloxacin','Verapamil','Ofloxacin','Tramadol',
             'Cypermethrine','Ibuprofen','1-chlorodecane','ButyltinTrichloride','Imidacloprid','Quinoxyfen',
             'Chlorothanolil','Isodrin','Dicofol','AzinphosMethyl','Diazinon','FentinChloride','MC-LR','Methanol']

if __name__ == '__main__':
    
    specie = {'E':'Erpobdella','G':'Gammarus','R':'Radix'}
    
    m = 'Tramadol'
    df = pd.read_csv(r'D:\VP\Viewpoint_data\REGS\Molecules\{}.csv'.format(m),index_col=None)
    concentrations = df['Concentration'].unique()
    print(concentrations)
    
    #conc = input('What concentration?')
    conc = '3mg'
    df_ = df[df['Concentration'] == conc]
    
    for i in range(df_.shape[0]):
        root = df_.iloc[i]['Erpobdella']
        root = r'I:\{}\{}'.format(root.split('\\')[1],root.split('\\')[2])
        
        try:
            data = TOX.csvDATA(root)
            TOX.ToxPLOT(data).plotIGT()
            
        except:
            print('No data yet for ',root)
            continue