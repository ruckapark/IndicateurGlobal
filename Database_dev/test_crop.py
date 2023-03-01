# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 09:40:44 2023

Test crop function on existing videos
Testvids should be directory containing at least one video
Corresponds to 'dopage'

@author: George
"""

import datetime
import toxprints_cropvids as crop

dopage = datetime.datetime.strptime('20220729 12:18:23', '%Y%m%d %H:%M:%S')
testvids = r'C:\Users\George\Documents\TestVids\2022'

crop.crop_videos(dopage,crop_hours = 2,directory = testvids,delete = True)