# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 11:13:53 2023

Plot all the final versions of datasets for each species# -*- coding: utf-8 -*-
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
    
    #Define species and root directory
    specie = {'E':'Erpobdella','G':'Gammarus','R':'Radix'}
    root_dir = r'D:\VP\Viewpoint_data\REGS\Molecules'
    
    #Define required molecule
    #m = input('Which moleucle?')
    m = 'Verapamil'
    
    #read selected directories
    df = pd.read_csv(r'D:\VP\Viewpoint_data\REGS\Molecules\{}_custom.csv'.format(m),index_col = 'Repetition')
    
    #for 3 species plot each IGT curve and label the associated repetition
    fig,axe = plt.subplots(1,3,figsize = (18,8),sharex = True)
    for i,s in enumerate(specie): 
        axe[i].set_title(specie[s])
        axe[i].set_ylim((-2,2))
    
    for i in range(df.shape[0]):
        entry = {s:df[specie[s]] for s in specie}
        data = TOX.csvDATA_comp({s:df[specie[s]].iloc[i] for s in specie})
        for x,s in enumerate(data.species):
            axe[x].plot(data.IGT[s],label = i)
    
    axe[x].legend()