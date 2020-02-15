import pandas as pd
import numpy as np
import os, sys, subprocess

df = pd.read_csv('human_labelled_candidates.csv')
df["Beam"] = df.Beam.map("{:02}".format)
print(df['Classification'].value_counts())

pulsars = df.loc[(df['Classification'] == 1) | (df['Classification'] == 2)]

non_pulsars = df.loc[(df['Classification'] == 0) | (df['Classification'] == 3)]
for index, row in pulsars.iterrows():
    cmds = 'cp /fred/oz002/vishnu/LOWLAT/candidate_plots_black_hole/'+ row['Pointing'] + '/' + row['Beam'] + '/' + row['ImageName'][:-4] + '* ' + '/fred/oz002/vishnu/neural_network/lowlat_cands/pulsars/'
    subprocess.check_output(cmds, shell=True)


for index, row in non_pulsars.iterrows():
    cmds = 'cp /fred/oz002/vishnu/LOWLAT/candidate_plots_black_hole/'+ row['Pointing'] + '/' + row['Beam'] + '/' + row['ImageName'][:-4] + '* ' + '/fred/oz002/vishnu/neural_network/lowlat_cands/nonpulsars/'
    subprocess.check_output(cmds, shell=True) 
