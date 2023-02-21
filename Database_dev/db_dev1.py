# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 11:02:19 2023

create database for ToxPrints log - start by reading in 

@author: George
"""

import sys
sys.path.insert(0, r'D:\VP\Viewpoint_data\code')
import MODS.dope_reg as dope_reg

if __name__ == '__main__':
    
    #read in dope reg to start with
    reg = dope_reg.dope_read()
    
    #find corresponding file roots
    
    
    #save to new dope reg in db folder
    """
    this should eventually contain everything not only weekly dopage
    """