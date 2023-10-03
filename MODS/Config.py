# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 10:31:53 2022

Class to read zones
Coordinates in image start in the top left.

@author: GRuck
"""

import os
import cv2
import csv
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
            
def write_mapping(config,filedate,Tox):
    
    #dictionary to rows
    rows = [[key]+config[key] for key in config]
    
    #write txt file with mapping. File name corresponds to date of config.
    with open(r'I:\Shared\Configs\Mappings\{}\{}.csv'.format(Tox,filedate), 'w', newline='') as f:
     
        # using csv.writer method from CSV package
        write = csv.writer(f)
        write.writerows(rows)
        
def read_mapping(Tox,rootdate = 20220225):
    
    #configroot i.e. r'I:\Shared\Configs\Mappings\769'
    #rootdate of datafile not mapping i.e. 20210702 from I:\TXM762-PC\20210702-082335
    #rootdate unless specified assumed tobe recent
    
    mapping = {}
    configroot = r'I:\Shared\Configs\Mappings\{}'.format(Tox)
    mappings = os.listdir(configroot)
    configpath = None
    
    if len(mappings) == 1:
        configpath = r'{}\{}'.format(configroot,mappings[0])
                
    else:        
        #loop from most recent date
        for m in mappings[::-1][1:]:
            mapdate = int(m.split('.')[0])
            if int(rootdate) > mapdate:
                configpath = r'{}\{}'.format(configroot,m)
                break
            
    if not configpath: configpath = r'{}\{}'.format(configroot,mappings[0])
                
    #read three line csv file containing true mapping order
    with open(configpath, 'r') as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            mapping.update({row[0]:[int(x) for x in row[1:]]})
                
    return mapping
            

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
        current,old = None,None
        
        #config
        for i,conf in enumerate(os.listdir(base)[::-1]):
            config = Config(r'{}\{}'.format(base,conf))
            if i:
                old = current
            current = config.mapping
                
            if config.mapping != nomap:
                print(Tox)
                print('\n')
                print(config.mapping)
                print('\n')
            
            #write most most recent mapping (else) or a changed mapping (if)
            if i:
                if old != current: write_mapping(current,conf.split('-')[0],Tox)
            else:
                write_mapping(current,int(conf.split('-')[0]),Tox)
              