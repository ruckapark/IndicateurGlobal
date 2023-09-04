#ruckapark
#movefile and zip

import os
import shutil
import zipfile

#ADD 2 inputs each time
input_dir = r'C:\Users\TXM762\Documents\ReplayVideos\762_20211022'
input_file = '20230614-172708.xls'

os.chdir(input_dir)
zip_file = r'{}.zip'.format(input_file)

with zipfile.ZipFile(zip_file,"w",zipfile.ZIP_DEFLATED) as archive:
    archive.write(input_file)


#locate correct output in I drive and use shutil to copy to this location


[Tox,direc] = input_dir.split('\\')[-1].split('_')
base_out = r'I:\TXM{}-PC'.format(Tox)
dir_out = [d for d in os.listdir(base_out) if direc in d][0]
output_dir = r'{}\{}'.format(base_out,dir_out)
shutil.move(zip_file,r'{}\{}.replay.xls.zip'.format(output_dir,dir_out))
