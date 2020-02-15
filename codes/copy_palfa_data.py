import glob, os, sys
import numpy as np
import subprocess

original_files_directory = '/beegfs/vishnu/final_test_set_numpy/'
filename = 'pfd_correct.txt' 
data = np.loadtxt(filename, dtype=[('fname', '|S200'), ('Overall', int), ('MJD', int)], comments='#')
os.chdir(original_files_directory)
output = []
for i in range(len(data['Overall'])):
    #print '%s file is being copied now.' %str(i+1) 
    if data['fname'][i].startswith('./'):
        files = data['fname'][i][2:]
        files = os.path.basename(files)
        if files.endswith('.pfd'):
            files = files[:-4]
            files = files + '.npy'
            if not os.path.isfile(files):
                 print files + ' does not exist'
            
            if data['Overall'][i]==2:
                print files
                data['Overall'][i] = 1

            output.append([files, str(data['Overall'][i])])


    else:
        files = data['fname'][i]
        files = os.path.basename(files)
        if files.endswith('.pfd'):
            files = files[:-4]
            files = files + '.npy'

            if not os.path.isfile(files):
                print files + ' does not exist'
            
            if data['Overall'][i]==2:

                data['Overall'][i] = 1

            output.append([files, str(data['Overall'][i])])



output = np.asarray(output)

if not os.path.exists('test_data_labels.txt'):
    with open('test_data_labels.txt', 'w') as outfile:
        np.savetxt(outfile, output, delimiter=',', fmt='%s')
