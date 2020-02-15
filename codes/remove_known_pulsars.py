import pandas as pd
import numpy as np
import subprocess

df1 = pd.read_csv('/fred/oz002/vishnu/LOWLAT/candidate_plots_black_hole/nearby_pulsars_htru_low_lat_pointings.csv')
df1["Beam"] = df1.Beam.map("{:02}".format)
df = pd.read_csv('all_candidates_above_50_percent.csv')
df["beam"] = df.beam.map("{:02}".format)
df['filename_pics'] = df['filename_pics'].astype(str) + '.png'
new_stuff = []
for index, row in df.iterrows():
#            #df4 = df.loc[(df['pics_score_palfa'] >= 0.90) & (df['pics_score_palfa'] < 0.70)]
#            df4 = df.loc[(df['pics_score_palfa'] >= 0.90)]
    known_pulsars = df1.loc[(df1['Pointing'] == row['pointing']) & (df1['Beam'] == row['beam']) & (df1['RAD.DISTANCE'] <= 1.0)]
    if known_pulsars.empty:
        new_stuff.append(False)
    else:
        known_pulsars_spin_period = known_pulsars['P0'].values
        known_pulsar_dm = known_pulsars['DM'].values
        cand_spin_period = row['topo_p1']
        cand_dm = row['best_dm']
        spin_ratio = cand_spin_period/known_pulsars_spin_period
        dm_ratio = cand_dm/known_pulsar_dm
        for j in range(len(spin_ratio)):
            if spin_ratio[j] >= 0.95 and spin_ratio[j] <= 1.05 and dm_ratio[j] >= 0.90 and dm_ratio[j]<=1.10:
                #print(cand_spin_period, cand_dm, spin_ratio[j], dm_ratio[j])
                pass
            else:

                cmds = 'cp ' + '/fred/oz002/vishnu/LOWLAT/candidate_plots_black_hole/' + row['pointing'] + '/' + row['beam'] + '/' +  row['filename_pics'] + ' ' + '/fred/oz002/vishnu/neural_network/above_50_percent_new_stuff_unclassified/'
#                        cmds = 'cp' + ' ' + dir_name + '/' + beam_name + '/' +  plots[i] + '*' + ' ' + '/fred/oz002/vishnu/LOWLAT/candidate_plots_black_hole/redetections/'
                subprocess.check_output(cmds, shell=True)
#                        tag=1
#
#                for factor in np.arange(1.0, 17.0):
#                         #Fractional Harmonics
#                    if np.fabs(cand_spin_freq - known_pulsars_spin_period[j]*factor) < f_err*factor and dm_ratio[j] >= 0.90 and dm_ratio[j]<=1.10:
#                            tag=1
#                            #cmds = 'cp' + ' ' + dir_name + '/' + beam_name + '/' +  plots[i] + '.png' + ' ' + '/fred/oz002/vishnu/LOWLAT/candidate_plots_black_hole/harmonics/'
#                            cmds = 'cp' + ' ' + dir_name + '/' + beam_name + '/' +  plots[i] + '*' + ' ' + '/fred/oz002/vishnu/LOWLAT/candidate_plots_black_hole/harmonics/'
