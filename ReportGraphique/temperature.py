# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 07:06:49 2021

@author: Admin
"""


import os
import pandas as pd
import matplotlib.pyplot as plt

os.chdir(r'D:\VP\Viewpoint_data')
file = 'temp_tox.csv'
temps = pd.read_csv(file)
temps = temps[temps.columns[1:4]]
temps = temps.dropna()
temps = temps.rename(columns = {'Unnamed: 1':'date'})
print(temps.head())
temps['date'] = pd.to_datetime(temps['date'],format = '%d/%m/%Y %H:%M:%S')
temps['TXM 764'] = temps['TXM 764'].apply(lambda x: float(x.split()[0]))
temps['TXM 768'] = temps['TXM 768'].apply(lambda x: float(x.split()[0]))
temps = temps.set_index(['date'])
temps = temps.iloc[1300:19000]
fig = plt.figure()
axe = fig.add_axes([0.1,0.3,0.8,0.6])
axe.set_title('Temperature plots 08-15 / 03')
plt.plot(temps.index,temps['TXM 768'],label = 'TXM 768')
plt.plot(temps.index,temps['TXM 764'],c = 'r',label = 'TXM 764')
axe.legend()