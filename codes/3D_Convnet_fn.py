import sys, os, glob
sys.path.append('/home/psr/software/psrchive/install/lib/python2.7/site-packages')
sys.path.append('/home/psr')
import numpy as np
from ubc_AI.training import pfddata
from samples import downsample, normalize
import subprocess

pfd_files_new_batch_nonpulsars = sorted(glob.glob('/fred/oz002/vishnu/neural_network/lowlat_cands/new_batch_nonpulsars/*.pfd')) + sorted(glob.glob('/fred/oz002/vishnu/neural_network/lowlat_cands/nonpulsars/*.pfd'))


def greyscale(img):
            global_max = np.maximum.reduce(np.maximum.reduce(img))
            min_parts = np.minimum.reduce(img, 1)
            img = (img-min_parts[:,np.newaxis])/global_max
            return img

#def 3d_convnet():


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

batch_number=0
for value in chunks(pfd_files_new_batch_nonpulsars,6000):

    batch_number+=1


    # Initialise data objects from getdata class
    data_object_new_batch_nonpulsars = [pfddata(f) for f in value]

    # Read the 3D Data in time, freq, phase format

    full_3d_array = [f.profs for f in data_object_new_batch_nonpulsars]
    grey_scale_3d_array = [greyscale(f) for f in full_3d_array]
    #No aligning done
    final_array = [normalize(downsample(f, 64))  for f in grey_scale_3d_array]
    np.save('/fred/oz002/vishnu/neural_network/lowlat_cands/full_3D_pfd_data_non_pulsars_batch_%d.npy' %int(batch_number), final_array)

model = Sequential()
model.add(Conv3D(32, kernel_size=(3, 3, 1), activation='relu', input_shape=input_shape))
model.add(Conv3D(32, kernel_size=(3, 3, 1), activation='relu'))
model.add(MaxPooling3D(pool_size=(2, 2, 1)))

model.add(Dropout(0.25))
model.add(Conv3D(64, kernel_size=(3, 3, 1), activation='relu'))
model.add(Conv3D(64, kernel_size=(3, 3, 1), activation='relu'))
model.add(MaxPooling3D(pool_size=(2, 2, 1)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))
