# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 18:06:45 2024

Mainly for the purpose of the manuscript, all figures are saved in Article 3 with BSpline plots

@author: George
"""

import os
import LAB_splines as SPL

if __name__ == "__main__":
    
    input_directory = r'D:\VP\ARTICLE3\ArticleData'  #find data means or IGTs
    IGT = [f for f in os.listdir(input_directory) if 'IGT' in f]
    mean = [f for f in os.listdir(input_directory) if 'mean' in f]
    
    for i in range(len(IGT)):
        
        IGT_s = SPL.ToxSplines(r'{}\{}'.format(input_directory,IGT[i]))
        mean_s = SPL.ToxSplines(r'{}\{}'.format(input_directory,mean[i]))
        
        #define image titel as PNG file without the csv extension in class title
        file_IGT = r'D:\VP\ARTICLE3\ArticleData\Figures\{}.PNG'.format(IGT_s.title.split('.')[0])
        file_mean = r'D:\VP\ARTICLE3\ArticleData\Figures\{}.PNG'.format(mean_s.title.split('.')[0])
        
        IGT_s.plot_raw(fname = file_IGT)
        mean_s.plot_raw(fname = file_mean)