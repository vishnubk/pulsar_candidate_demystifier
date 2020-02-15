import numpy as np
import pandas as pd
import sys, os, glob, subprocess
from sklearn.model_selection import train_test_split

pfd_files_pulsars = sorted(glob.glob('lowlat_cands/pulsars/*.pfd')) + sorted(glob.glob('lowlat_cands/new_batch_pulsars/*.pfd'))
pfd_files_nonpulsars = sorted(glob.glob('lowlat_cands/nonpulsars/*.pfd')) + sorted(glob.glob('lowlat_cands/new_batch_nonpulsars/*.pfd'))
#with open ('lowlat_cands_labelled.txt', 'w') as f:
#   for filename in pfd_files_pulsars:
#       f.write('/fred/oz002/vishnu/neural_network/' + filename + ' ' + '1' + ' ' + '1' + ' ' + '1' + ' ' + '1' + ' ' + '1' + '\n')
#   for filename in pfd_files_nonpulsars:
#       f.write('/fred/oz002/vishnu/neural_network/' + filename + ' ' + '0' + ' ' + '0' + ' ' + '0' + ' ' + '0' + ' ' + '0' + '\n')
df = pd.read_csv('lowlat_cands_labelled.txt', header=None, sep = ' ', names = ['filename', 'Overall','Pulsar_Profile','DM_Curve','Time_Phase','Freq_Phase'])
data = df['filename'].tolist()
label = df['Overall'].tolist()

#df['filename'] = df['filename'].str.replace('/var/tmp/jobfs/data_vishnu/'
print(df['filename'].str.split('/',expand=True))

#train, test = train_test_split(df, test_size=0.3, random_state=42)
#print(train)
#train.to_csv('lowlat_cands_train.txt', sep = ' ', header=False,index=False)
#test.to_csv('lowlat_cands_test.txt', sep = ' ', header=False,index=False)
