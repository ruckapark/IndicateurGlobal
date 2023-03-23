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

dopage = datetime.datetime.strptime('20201211 10:01:00', '%Y%m%d %H:%M:%S')
testvids = r'I:\TXM763-PC\20201211-080603'

crop.crop_videos(dopage,crop_hours = 2,directory = testvids,delete = False)