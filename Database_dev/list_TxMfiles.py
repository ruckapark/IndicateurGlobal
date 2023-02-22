import os

#written in I drive for quick list of file s: Python 2.7
if __name__ == '__main__':

    #create txt file (csv) with all experiment names
    with open('allfiles.txt','w') as f:
        
        dirs = [d for d in os.listdir(os.getcwd()) if 'TXM' in d]
        for directory in dirs:
            #list all experiment directories
            os.chdir(directory)
            TxM = int(directory[3:6])
            exps = [e for e in os.listdir(os.getcwd()) if len(e) == 15]
            
            #write filenames to txt file
            for e in exps: f.write('{},{}\n'.format(e,TxM))
            
            #return to parent dir
            os.chdir('..')