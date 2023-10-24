# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 11:26:33 2023

Generate IGT and meandata for dataset2

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

os.chdir('MODS')
from data_merge import merge_dfs
from dope_reg import dope_read_extend
import dataread as d_
os.chdir('..')

#%% data registers
coppers = [
    r'I:\TXM765-PC\20210422-111620',
    r'I:\TXM767-PC\20210430-124553',
    r'I:\TXM767-PC\20210513-231929',
    r'I:\TXM763-PC\20210528-113951'
    ]

zincs = [
    r'I:\TXM763-PC\20210416-113757',
    r'I:\TXM763-PC\20210506-230746',
    r'I:\TXM763-PC\20210513-230658',
    r'I:\TXM763-PC\20210520-224858'
    ]

methomyls = [
    r'I:\TXM760-PC\20210520-224501',
    r'I:\TXM760-PC\20210625-093621',
    r'I:\TXM761-PC\20210520-224549',
    r'I:\TXM761-PC\20210625-093641'
    ]

tramadols = [
    r'I:\TXM767-PC\20220225-091008',
    r'I:\TXM768-PC\20220225-090953',
    r'I:\TXM769-PC\20220310-113807',
    r'I:\TXM769-PC\20220317-164759']

#deal with class to only account for active species
datasets = coppers + zincs + methomyls

if __name__ == "__main__":
    
    for r in datasets:
        data = TOX.csvDATA(r)
        data.write_data(r'D:\VP\ARTICLE2\ArticleData')