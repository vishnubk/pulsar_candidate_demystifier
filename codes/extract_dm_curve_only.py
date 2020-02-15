import sys, os, glob
sys.path.append('/home/psr/software/psrchive/install/lib/python2.7/site-packages')
sys.path.append('/home/psr')
import numpy as np
from ubc_AI.training import pfddata
from samples import downsample, normalize
import subprocess

#pfd_files_new_batch_nonpulsars = sorted(glob.glob('/fred/oz002/vishnu/neural_network/lowlat_cands/new_batch_pulsars/*.pfd')) + sorted(glob.glob('/fred/oz002/vishnu/neural_network/lowlat_cands/pulsars/*.pfd'))
#
#
#def greyscale(img):
#            global_max = np.maximum.reduce(np.maximum.reduce(img))
#            min_parts = np.minimum.reduce(img, 1)
#            img = (img-min_parts[:,np.newaxis])/global_max
#            return img
#
#
#
#def chunks(l, n):
#    """Yield successive n-sized chunks from l."""
#    for i in range(0, len(l), n):
#        yield l[i:i + n]
#
#batch_number=0
#for value in chunks(pfd_files_new_batch_nonpulsars,6000):
#
#    batch_number+=1
#
#
#    # Initialise data objects from getdata class
#    data_object_new_batch_nonpulsars = [pfddata(f) for f in value]
#
#    # Read the 3D Data in time, freq, phase format
#
#    full_3d_array = [f.profs for f in data_object_new_batch_nonpulsars]
#    grey_scale_3d_array = [greyscale(f) for f in full_3d_array]
#    #No aligning done
#    final_array = [normalize(downsample(f, 64))  for f in grey_scale_3d_array]

#    np.save('/fred/oz002/vishnu/neural_network/lowlat_cands/full_3D_pfd_data_pulsars_batch_%d.npy' %int(batch_number), final_array)


pfd_files_new_batch_nonpulsars = sorted(glob.glob('/fred/oz002/vishnu/neural_network/lowlat_cands/new_batch_pulsars/*.pfd')) + sorted(glob.glob('/fred/oz002/vishnu/neural_network/lowlat_cands/pulsars/*.pfd')) + sorted(glob.glob('/fred/oz002/vishnu/neural_network/lowlat_cands/nonpulsars/*.pfd')) + sorted(glob.glob('/fred/oz002/vishnu/neural_network/lowlat_cands/new_batch_nonpulsars/*.pfd'))
#pfd_files_new_batch_nonpulsars = sorted(glob.glob('/fred/oz002/vishnu/neural_network/lowlat_cands/pulsars/*.pfd')) 

#dm_curve = sorted(glob.glob('/fred/oz002/vishnu/neural_network/data/dm*.npy')) 
#dm_curve = [os.path.basename(f) for f in dm_curve]
#dm_curve_prefix_removed = [f[9:] for f in dm_curve]
#pfd_files_new_batch_nonpulsars = [os.path.basename(f) for f in pfd_files_new_batch_nonpulsars]
#pfd_files_new_batch_nonpulsars_w_postfix = [f[:-4] + '.npy' for f in pfd_files_new_batch_nonpulsars]
##print(pfd_files_new_batch_nonpulsars_w_postfix[0:10])
##print(dm_curve_prefix_removed[0:10])
##def greyscale(img):
##            global_max = np.maximum.reduce(np.maximum.reduce(img))
##            min_parts = np.minimum.reduce(img, 1)
##            img = (img-min_parts[:,np.newaxis])/global_max
##            return img
##
##
#remaining_files = [ f for f in pfd_files_new_batch_nonpulsars_w_postfix ] 
#for filename in dm_curve_prefix_removed: 
#    if filename in pfd_files_new_batch_nonpulsars_w_postfix: 
#        remaining_files.remove(filename)
  #else:
  #    print(filename)

for i in range(len(pfd_files_new_batch_nonpulsars)):
#for i in range(len(remaining_files)):    
 #   print(remaining_files[i][:-4] + '.pfd')
    # Initialise data objects from getdata class
     if not os.path.isfile('/fred/oz002/vishnu/neural_network/data/' + 'dm_curve_' + os.path.basename(pfd_files_new_batch_nonpulsars[i])[:-4] + '.npy'):
         data_object_new_batch_nonpulsars = pfddata(pfd_files_new_batch_nonpulsars[i])
##
##    # Read the 3D Data in time, freq, phase format
##
         dm_curve = data_object_new_batch_nonpulsars.getdata(DMbins=60) 
#    #No aligning done
#    
         np.save('/fred/oz002/vishnu/neural_network/data/' + 'dm_curve_' + os.path.basename(pfd_files_new_batch_nonpulsars[i])[:-4] + '.npy', dm_curve)
