#ruckapark
#movefile and zip

# adapt to be used on all PCs, should move xls, phr and raw zips, as well as morts.csv
import os
import shutil
import zipfile

def read_roots(root):
    
    return [d for d in os.listdir(root) if '_202' in d]

def find_fileroot():
    
    [f for f in os.listdir() if '.phr' in f]
    
def zipfile():
    zip_file = r'{}.zip'.format(input_file)
    
    with zipfile.ZipFile(zip_file,"w",zipfile.ZIP_DEFLATED) as archive:
        archive.write(input_file)

if __name__ == '__main__':
    
    root = r'C:\Users\George\Documents\ReplayTxM_files\763'
    roots = read_roots(root)
    
    for r in roots:
        
        print(r)
        input_dir = r'{}\{}'.format(root,r)
        os.chdir(input_dir)
        
        input_file = [f for f in os.listdir() if '.phr' in f]
        if len(input_file) != 1:
            print('Check root:', r)
            continue
        else:
            input_file = input_file[0].split('.')[0]
        
        #locate correct output in I drive and use shutil to copy to this location
        [Tox,direc] = input_dir.split('\\')[-1].split('_')
        base_out = r'I:\TXM{}-PC'.format(Tox)
        dir_out = [d for d in os.listdir(base_out) if direc in d][0]
        output_dir = r'{}\{}'.format(base_out,dir_out)
        
        for extension in ['.phr.zip','.raw.zip','.xls.zip']:
            f = r'{}\{}{}'.format(input_dir,input_file,extension)
            if not os.path.isfile(r'{}\{}.replay.{}'.format(output_dir,dir_out,extension)):
                try:
                    shutil.move(f,r'{}\{}.replay.{}'.format(output_dir,dir_out,extension))
                except:
                    print('Check zipfiles: ',output_dir)
                    continue