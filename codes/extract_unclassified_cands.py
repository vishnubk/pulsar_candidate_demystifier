import pandas as pd
import numpy as np
import subprocess
df = pd.read_csv('human_labelled_candidates.csv')
df1 = pd.read_csv('all_candidates_above_50_percent.csv')
df1['filename_pics'] = df1['filename_pics'].astype(str) + '.png'
df1["beam"] = df1.beam.map("{:02}".format)


df3 = df1.merge(df.drop_duplicates(), left_on = 'filename_pics', right_on = 'Cand Name', how='left', indicator=True)
df4 = df3[df3["_merge"] == "left_only"].drop(["_merge", "Proj. Radius", "P_orb", "Cand Name", "Spin Period", "DM", "PICS Score",  "TimeStamp", "Classification", "Pointing", "Beam", "UserName", "ImageName"], axis=1)

df4.to_csv('remaining_cands_still_to_be_labelled_above_50_percent.csv', index=False)
for index, row in df4.iterrows():
    cmds = 'cp /fred/oz002/vishnu/LOWLAT/candidate_plots_black_hole/'+ row['pointing'] + '/' + row['beam'] + '/' + row['filename_pics'] + ' ' + '/fred/oz002/vishnu/neural_network/unclassified_cands/'
    subprocess.check_output(cmds, shell=True)


#df4 = df3[df3["_merge"] == "left_only"].drop(columns=["_merge"])
#df4 = df3[df3["_merge"] == "left_only"]
#print(df.columns)
#print(df1.columns)
#print(df['Pointing'])
#print(df4.head())
#df4.columns = df.columns

#df = df4
#print(df1.columns, df.columns)
