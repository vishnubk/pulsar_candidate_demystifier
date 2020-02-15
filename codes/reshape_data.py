import numpy as np
import pandas as pd
import os
df = pd.read_csv('lowlat_cands_full_labelled.csv', header=None, names = ['filename', 'label']) 
for index, row in df.iterrows():
    filename = 'data/' + row['filename']
    data = np.load(filename)
    data = np.reshape(data, (64,64,64,1))
    np.save('data/reshaped_' + os.path.basename(filename), data)
