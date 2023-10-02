# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 10:31:53 2022

Class to read zones
Coordinates in image start in the top left.

@author: GRuck
"""

import os
import cv2
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

#for version 1 assume that the required config is known before calling class
class Config:
    
    def __init__(self,configname):
        
        self.configname = configname
        
        self.tree = ET.parse(self.configname)
        self.root = self.tree.getroot()
        
        self.get_camMat()
        self.get_coordinates()
        self.get_mapping()
        
    def get_camMat(self):
        """ Return Camera matrix coeffecients for each species"""
        self.camMat = {}
        self.distCoef = {}
        
        self.species = []
        for editor in self.root.iter('Set'):
            for att in editor.iter('Attribut'):
                if att.get('Name') == 'Location':
                    if att.get('Value')[0] in self.species:
                        break #only out of one loop
                    else:
                        sp = att.get('Value')[0]
                        self.species.append(sp)
                if att.get('Name') == 'CalibrationCameraMatrix':
                    camMat = att.get('Value').split(',')[:-1]
                    camMat = np.array([camMat[:3],camMat[3:6],camMat[6:]],dtype = float)
                    self.camMat.update({sp:camMat})
                elif att.get('Name') == 'CalibrationDistCoeffs':
                    distCoef = np.array(att.get('Value').split(',')[:-1],dtype = float)
                    self.distCoef.update({sp:distCoef})
                
    
    def get_coordinates(self):
        """
        Find coordinates of config for each species
        """
        self.points = {}
            
        flux = []
        for editor in self.root.iter('Editor'):
            cadres = []
            for polygon in editor.iter('polygon'):
                coords = polygon.get('points').split()
                points = [[int(coord.split(',')[0]),int(coord.split(',')[1])] for coord in coords]
                cadres.append(points)
            flux.append(cadres[0:16]) #only add first 16 (32 include dupliates)
            
        for i,sp in enumerate(self.root.iter('Alias')):
            self.points.update({sp.attrib['Name'][0].upper():np.array(flux)[i]})
            
    def get_mapping(self):
        """ 
        Find correct order of cells to avoid confusion
        If cells have been traced in correct order, the mapping operation has no effect
        
        Seperate all points into 4 rows and 4 columns
        Find each value for zone and rank zones 1-16
        """
        self.mapping = {}
        
        #find mapping
        for key in self.points:
            xrows = np.array([np.min(self.points[key][i][:,0]) for i in range(len(self.points[key]))])
            yrows = np.array([np.min(self.points[key][i][:,1]) for i in range(len(self.points[key]))])
            xrow1,xrow2,xrow3,xrow4 = np.sort(xrows)[:4],np.sort(xrows)[4:8],np.sort(xrows)[8:12],np.sort(xrows)[12:]
            yrow1,yrow2,yrow3,yrow4 = np.sort(yrows)[:4],np.sort(yrows)[4:8],np.sort(yrows)[8:12],np.sort(yrows)[12:]
            
            zones = []
            for i in range(len(self.points[key])):
                x = self.find_row(xrows[i],[xrow1,xrow2,xrow3,xrow4])
                y = self.find_row(yrows[i],[yrow1,yrow2,yrow3,yrow4])
                zones.append(1 + (4*y + x))
            
            self.mapping.update({key:zones})
        
    def find_row(self,coord,rows):
        for i in range(len(rows)):
            if coord in rows[i]:
                return i
            
    def plot_zones(self,sp):
        """ For given species plot the grid of zones in order """
        fig,ax = plt.subplots()    
        ax.set_xlim([0,964])
        ax.set_ylim([0,1288])
        ax.invert_yaxis()
        for i in range(len(self.points[sp])):
            true_index = self.mapping[sp].index(i+1)
            ax.add_patch(Polygon(self.points[sp][true_index]))
            plt.text(np.mean(self.points[sp][true_index][:,0]),np.mean(self.points[sp][true_index][:,1]),i+1)
          
if __name__ == "__main__":
    
    """
    os.chdir(r'D:\VP\Viewpoint_data\RAW\TestData')
    conf = Config('TEST_DefaultConfig.xml')
    
    sp = 'E'
    
    conf.plot_zones(sp)
    """
    
    nomap = {'E':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],'G': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],'R':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]}
    
    for Tox in range(760,770):
    #Tox = 763
        
        #base
        base = r'I:\Shared\Configs\Record\{}'.format(Tox)
        
        #config
        for conf in os.listdir(base)[::-1]:
            config = Config(r'{}\{}'.format(base,conf))
            if config.mapping != nomap:
                print(Tox)
                print('\n')
                print(config.mapping)
                print('\n')
        
        #Analysis shows this has never changed
        