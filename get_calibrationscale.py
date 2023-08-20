# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 17:55:10 2023

Specific to given directory

@author: George
"""

import xml.etree.ElementTree as ET
import pandas as pd
import os

if __name__ == "__main__":
    
    #file to write to from df
    file = r'D:\VP\Viewpoint_data\REGS\calibrationscales.csv'
    root = r'I:\Shared\Configs\DEEP'
    
    #read each config from Tox's in loop and extract calibration scale
    # loop through these modules AcqDirectShowVideo
    
    calibrationscales = {'Tox':[],'Date':[],'Erpobdella':[],'Gammarus':[],'Radix':[]}
    
    os.chdir(root)
    for d in os.listdir(root):
        
        Tox = int(d)
        os.chdir(d)
        
        for config in os.listdir(os.getcwd()):
            
            xml = ET.parse(config)
            
            calibrationscales['Tox'].append(Tox)
            calibrationscales['Date'].append(config[:8])
            
            for i,el in enumerate(xml.findall(".//Module[@Name='AcqDirectShowVideo']")):
                
                spec = el.findall(".//Alias")[0].attrib['Name']
                scale = el.findall(".//Attribut[@Name='Calibration']")[0].attrib['Value']
                scale = float(scale.split('scale=')[-1])
                calibrationscales[spec].append(scale)
                
        os.chdir(root)
        
    scales = pd.DataFrame(calibrationscales)
    
    #check for false scales (should be at least 2)
    for i in range(scales.shape[0]):
        if scales.loc[i][2:].min() < 2.0: scales = scales.drop(axis = 0,index = i)
        
    scales = scales.reset_index(drop = True)
    
    scales.to_csv(file,header = False,index = False)