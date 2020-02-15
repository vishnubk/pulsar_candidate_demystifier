
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from keras import backend as K
from keras import optimizers
import numpy as np
import math, time
import itertools
from sklearn.model_selection import train_test_split
from keras.callbacks import TensorBoard

# Load all data
time_phase_pulsars = np.load('input_data/time_phase_data_pulsars.npy')
time_phase_nonpulsars = np.load('input_data/time_phase_data_nonpulsars.npy')

freq_phase_pulsars = np.load('input_data/freq_phase_data_pulsars.npy')
freq_phase_nonpulsars = np.load('input_data/freq_phase_data_nonpulsars.npy')

pulse_profile_pulsars = np.load('input_data/pulse_profile_data_pulsars.npy')
pulse_profile_nonpulsars = np.load('input_data/pulse_profile_data_nonpulsars.npy')

dm_curve_pulsars = np.load('input_data/dm_curve_data_pulsars.npy')
dm_curve_nonpulsars = np.load('input_data/dm_curve_data_nonpulsars.npy')

pulse_profile_gbncc_pulsars = np.load('input_data/pulse_profile_gbncc_test_data_pulsars.npy')
pulse_profile_gbncc_nonpulsars = np.load('input_data/pulse_profile_gbncc_test_data_nonpulsars_part1.npy')

reshaped_time_phase_pulsars = [np.reshape(f,(48,48,1)) for f in time_phase_pulsars]
reshaped_time_phase_nonpulsars = [np.reshape(f,(48,48,1)) for f in time_phase_nonpulsars]

reshaped_freq_phase_pulsars = [np.reshape(f,(48,48,1)) for f in freq_phase_pulsars]
reshaped_freq_phase_nonpulsars = [np.reshape(f,(48,48,1)) for f in freq_phase_nonpulsars]

print('Total Number of Pulsar Examples is %d' %len(dm_curve_pulsars))
print('Total Number of Non-Pulsar Examples is %d' %len(dm_curve_nonpulsars))


