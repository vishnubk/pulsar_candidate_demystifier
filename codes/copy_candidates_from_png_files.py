import os, sys, subprocess, glob
#sys.path.append('/apps/skylake/software/core/anaconda3/5.1.0/lib/python3.6/site-packages')
import pandas as pd
import numpy as np
import psutil
def childCount():
    current_process = psutil.Process()
    children = current_process.children()
    return(len(children))

data = glob.glob('/fred/oz002/vishnu/neural_network/lowlat_cands/new_batch_nonpulsars/*.png')
data1 = pd.DataFrame(data,  columns = ['Cand Name'])
data1['Cand Name'] = data1['Cand Name'].apply(lambda x: os.path.basename(x))
df1 = pd.read_csv('all_candidates_above_20_percent_below_50_percent.csv')
df1['filename_pics'] = df1['filename_pics'].astype(str) + '.png'
df1["beam"] = df1.beam.map("{:02}".format)
#
#print(df1['filename_pics'][2])
#
df3 = df1.merge(data1, left_on = 'filename_pics', right_on = 'Cand Name', how='left', indicator=True)
#df4 = df3[df3["_merge"] == "left_only"].drop(["_merge", "Proj. Radius", "P_orb", "Cand Name", "Spin Period", "DM", "PICS Score",  "TimeStamp", "Classification", "Pointing", "Beam", "UserName", "ImageName"], axis=1)
#df4 = df3[df3["_merge"] == "left_only"].drop(["_merge", 'Cand Name'], axis=1)
df4 = df3[df3["_merge"] == "both"].drop(["_merge", 'Cand Name'], axis=1)
#print(df4['filename_pics'].drop_duplicates())
#print(df4['filename_pics'].tolist()[1])
#print(len(data))
#print(df4)
for index, row in df4.iterrows():
    cmds = 'cp /fred/oz002/vishnu/LOWLAT/candidate_plots_black_hole/'+ row['pointing'] + '/' + row['beam'] + '/' + row['filename_pics'][:-4] + '* ' + '/fred/oz002/vishnu/neural_network/lowlat_cands/new_batch_nonpulsars/'
    p = subprocess.Popen(cmds,shell=True)
    while childCount() > 25:
        time.sleep(10)
        p.communicate() #


#    subprocess.check_output(cmds, shell=True)
#test = df4['filename_pics'].tolist()
#print(df4['filename_pics'])
#print(len(test))
#print(df3)
#print([os.path.basename(f) for f in data])
#print(len(data1.index))
