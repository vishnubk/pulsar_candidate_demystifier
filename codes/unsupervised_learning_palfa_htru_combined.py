#!/usr/bin/env python
# coding: utf-8



import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import keras
from keras import regularizers
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers import Activation, Dropout, Flatten, Dense, LeakyReLU, Reshape
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, log_loss, accuracy_score
from keras import backend as K
from keras import optimizers
import numpy as np
import math, time
import itertools
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, concatenate
from keras.models import Model
from sklearn.cluster import KMeans
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
import glob
from my_classes import ClusteringLayer
from skimage.transform import resize
#get_ipython().run_line_magic('matplotlib', 'inline')

from keras.backend.tensorflow_backend import set_session  
config = tf.ConfigProto()  
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU  
sess = tf.Session(config=config)  
set_session(sess)
#time_phase_pulsars_combined = 

# Load all data
time_phase_pulsars_combined_array = np.empty(shape=(0,2304))
freq_phase_pulsars_combined_array = np.empty(shape=(0,2304))
pulse_profile_pulsars_combined_array = np.empty(shape=(0,64))
dm_curve_pulsars_combined_array = np.empty(shape=(0,60))



time_phase_pulsars_files = sorted(glob.glob('lowlat_cands/pulsars/time_phase_data_pulsars_batch_*.npy')) + sorted(glob.glob('lowlat_cands/new_batch_pulsars/time_phase_data_new_batch_pulsars_batch_*.npy'))
freq_phase_pulsars_files = sorted(glob.glob('lowlat_cands/pulsars/freq_phase_data_pulsars_batch_*.npy')) + sorted(glob.glob('lowlat_cands/new_batch_pulsars/freq_phase_data_new_batch_pulsars_batch_*.npy'))
pulse_profile_pulsars_files = sorted(glob.glob('lowlat_cands/pulsars/pulse_profile_data_pulsars_batch_*.npy')) + sorted(glob.glob('lowlat_cands/new_batch_pulsars/pulse_profile_data_new_batch_pulsars_batch_*.npy'))
dm_curve_pulsars_files = sorted(glob.glob('lowlat_cands/pulsars/dm_curve_data_pulsars_batch_*.npy')) + sorted(glob.glob('lowlat_cands/new_batch_pulsars/dm_curve_data_new_batch_pulsars_batch_*.npy'))


time_phase_pulsars_data = [np.load(f) for f in time_phase_pulsars_files]
freq_phase_pulsars_data = [np.load(f) for f in freq_phase_pulsars_files]
pulse_profile_pulsars_data = [np.load(f) for f in pulse_profile_pulsars_files]
dm_curve_pulsars_data = [np.load(f) for f in dm_curve_pulsars_files]

#shape = [np.concatenate((time_phase_pulsars_combined_array, f), axis = 0) for f in time_phase_pulsars_data]

for f in time_phase_pulsars_data:
    time_phase_pulsars_combined_array = np.concatenate((time_phase_pulsars_combined_array, f), axis = 0)

for f in freq_phase_pulsars_data:
    freq_phase_pulsars_combined_array = np.concatenate((freq_phase_pulsars_combined_array, f), axis = 0)


for f in pulse_profile_pulsars_data:
    pulse_profile_pulsars_combined_array = np.concatenate((pulse_profile_pulsars_combined_array, f), axis = 0)


for f in dm_curve_pulsars_data:
    dm_curve_pulsars_combined_array = np.concatenate((dm_curve_pulsars_combined_array, f), axis = 0)


time_phase_nonpulsars_combined_array = np.empty(shape=(0,2304))
freq_phase_nonpulsars_combined_array = np.empty(shape=(0,2304))
pulse_profile_nonpulsars_combined_array = np.empty(shape=(0,64))
dm_curve_nonpulsars_combined_array = np.empty(shape=(0,60))

#Load all files for non-pulsars
#time_phase_nonpulsars_files = ['lowlat_cands/nonpulsars/time_phase_data_nonpulsars.npy'] + sorted(glob.glob('lowlat_cands/new_batch_nonpulsars/time_phase_data_new_batch_nonpulsars_batch_*.npy'))
#freq_phase_nonpulsars_files = ['lowlat_cands/nonpulsars/freq_phase_data_nonpulsars.npy'] + sorted(glob.glob('lowlat_cands/new_batch_nonpulsars/freq_phase_data_new_batch_nonpulsars_batch_*.npy'))
#pulse_profile_nonpulsars_files = ['lowlat_cands/nonpulsars/pulse_profile_data_nonpulsars.npy'] + sorted(glob.glob('lowlat_cands/new_batch_nonpulsars/pulse_profile_data_new_batch_nonpulsars_batch_*.npy'))
#dm_curve_nonpulsars_files = ['lowlat_cands/nonpulsars/dm_curve_data_nonpulsars.npy'] + sorted(glob.glob('lowlat_cands/new_batch_nonpulsars/dm_curve_data_new_batch_nonpulsars_batch_*.npy'))


time_phase_nonpulsars_files = sorted(glob.glob('lowlat_cands/new_batch_nonpulsars/time_phase_data_new_batch_nonpulsars_batch_*.npy'))
freq_phase_nonpulsars_files = sorted(glob.glob('lowlat_cands/new_batch_nonpulsars/freq_phase_data_new_batch_nonpulsars_batch_*.npy'))
pulse_profile_nonpulsars_files = sorted(glob.glob('lowlat_cands/new_batch_nonpulsars/pulse_profile_data_new_batch_nonpulsars_batch_*.npy'))
dm_curve_nonpulsars_files = sorted(glob.glob('lowlat_cands/new_batch_nonpulsars/dm_curve_data_new_batch_nonpulsars_batch_*.npy'))

time_phase_nonpulsars_data = [np.load(f) for f in time_phase_nonpulsars_files]
freq_phase_nonpulsars_data = [np.load(f) for f in freq_phase_nonpulsars_files]
pulse_profile_nonpulsars_data = [np.load(f) for f in pulse_profile_nonpulsars_files]
dm_curve_nonpulsars_data = [np.load(f) for f in dm_curve_nonpulsars_files]

#shape = [np.concatenate((time_phase_pulsars_combined_array, f), axis = 0) for f in time_phase_pulsars_data]

for f in time_phase_nonpulsars_data:
    time_phase_nonpulsars_combined_array = np.concatenate((time_phase_nonpulsars_combined_array, f), axis = 0)


for f in freq_phase_nonpulsars_data:
    freq_phase_nonpulsars_combined_array = np.concatenate((freq_phase_nonpulsars_combined_array, f), axis = 0)


for f in pulse_profile_nonpulsars_data:
    pulse_profile_nonpulsars_combined_array = np.concatenate((pulse_profile_nonpulsars_combined_array, f), axis = 0)


for f in dm_curve_nonpulsars_data:
    dm_curve_nonpulsars_combined_array = np.concatenate((dm_curve_nonpulsars_combined_array, f), axis = 0)


print('Total Number of Pulsar Examples is %d' %dm_curve_pulsars_combined_array.shape[0])
print('Total Number of Non-Pulsar Examples is %d' %dm_curve_nonpulsars_combined_array.shape[0])
print('Total Labelled Data Available is %d'%(dm_curve_pulsars_combined_array.shape[0] + dm_curve_nonpulsars_combined_array.shape[0]))

# Reshape Arrays

reshaped_time_phase_pulsars = [np.reshape(f,(48,48,1)) for f in time_phase_pulsars_combined_array]
reshaped_time_phase_nonpulsars = [np.reshape(f,(48,48,1)) for f in time_phase_nonpulsars_combined_array]

reshaped_freq_phase_pulsars = [np.reshape(f,(48,48,1)) for f in freq_phase_pulsars_combined_array]
reshaped_freq_phase_nonpulsars = [np.reshape(f,(48,48,1)) for f in freq_phase_nonpulsars_combined_array]

#Load Palfa Data

time_phase_pulsars_palfa_data = np.load('palfa_data/time_phase_data_palfa_data_pulsars.npy')
freq_phase_pulsars_palfa_data = np.load('palfa_data/freq_phase_data_palfa_data_pulsars.npy')
pulse_profile_pulsars_palfa_data = np.load('palfa_data/pulse_profile_data_palfa_data_pulsars.npy')
dm_curve_pulsars_palfa_data = np.load('palfa_data/dm_curve_data_palfa_data_pulsars.npy')

time_phase_nonpulsars_palfa_data = np.load('palfa_data/time_phase_data_palfa_data_nonpulsars.npy')
freq_phase_nonpulsars_palfa_data = np.load('palfa_data/freq_phase_data_palfa_data_nonpulsars.npy')
pulse_profile_nonpulsars_palfa_data = np.load('palfa_data/pulse_profile_data_palfa_data_nonpulsars.npy')
dm_curve_nonpulsars_palfa_data = np.load('palfa_data/dm_curve_data_palfa_data_nonpulsars.npy')


print('Total Number of Palfa Pulsar Examples is %d' %dm_curve_nonpulsars_palfa_data.shape[0])
print('Total Number of Palfa Non-Pulsar Examples is %d' %freq_phase_pulsars_palfa_data.shape[0])
#Reshape Palfa Data

reshaped_time_phase_pulsars_palfa = [np.reshape(f,(48,48,1)) for f in time_phase_pulsars_palfa_data]
reshaped_time_phase_nonpulsars_palfa = [np.reshape(f,(48,48,1)) for f in time_phase_nonpulsars_palfa_data]

reshaped_freq_phase_pulsars_palfa = [np.reshape(f,(48,48,1)) for f in freq_phase_pulsars_palfa_data]
reshaped_freq_phase_nonpulsars_palfa = [np.reshape(f,(48,48,1)) for f in freq_phase_nonpulsars_palfa_data]

#
#
## Helper Functions 
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.5f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9
    
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    img_shape = 48, 48
    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.savefig('time_phase_examples.png')
    #plt.show()



#
img_width, img_height = 48, 48
input_shape = (48, 48, 1)
batch_size = 200
##tensor_board = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
##NAME = "Simple-CNN"
##tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
if K.image_data_format() == 'channels_first':
    input_shape = (1, img_width, img_height)
else:
    input_shape = (img_width, img_height, 1)

# Generator is used to avoid reading entire data into system memory.
def generator(batch_size,from_list_x,from_list_y):

    assert len(from_list_x) == len(from_list_y)
    total_size = len(from_list_x)

    while True:

        for i in range(0,total_size,batch_size):
            yield np.array(from_list_x[i:i+batch_size]), np.array(from_list_y[i:i+batch_size])


## In[9]:
#
#
def create_convolution_layers(input_img):
  model = Conv2D(32, (3, 3), padding='same', input_shape=input_shape)(input_img)
  model = LeakyReLU(alpha=0.1)(model)
  model = MaxPooling2D((2, 2),padding='same')(model)
  model = Dropout(0.25)(model)
  
  model = Conv2D(64, (3, 3), padding='same')(model)
  model = LeakyReLU(alpha=0.1)(model)
  model = MaxPooling2D(pool_size=(2, 2),padding='same')(model)
  model = Dropout(0.25)(model)
    
  model = Conv2D(128, (3, 3), padding='same')(model)
  model = LeakyReLU(alpha=0.1)(model)
  model = MaxPooling2D(pool_size=(2, 2),padding='same')(model)
  model = Dropout(0.4)(model)
  model = Flatten()(model)
  model = Dense(48)(model)
  model = LeakyReLU(alpha=0.1)(model)
  
  return model

def create_perceptron_layer(input_data):
    model = Dense(64, activation="relu")(input_data)
    model = Dense(32, activation="relu")(model)
    #model = Dense(16, activation="relu")(model)
   # model = Model(inputs=input_data, outputs=model)
    
    return model

def create_convolutional_autoencoder(input_img):

    x = Conv2D(16, (5, 5), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (5, 5), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (5, 5), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    #x = Conv2D(8, (4, 4), activation='relu', padding='same')(x)
    #x = MaxPooling2D((2, 2), padding='same')(x)
    encoded = x
    #encoded = MaxPooling2D((2, 2), padding='same')(x)
    encoded_results = Flatten()(encoded) 
    # at this point the representation is (4, 4, 8) i.e. 128-dimensional


    x = Conv2D(8, (5, 5), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    #x = UpSampling2D((3, 3))(encoded)
    #x = Conv2D(8, (4, 4), activation='relu', padding='same')(x)
    #x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (5, 5), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (5, 5), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    #x = Dropout(0.3)(x)
    decoded = Conv2D(1, (5, 5), activation='sigmoid', padding='same')(x)
    return Model(inputs=input_img, outputs=decoded, name='AE'), Model(inputs=input_img, outputs=encoded_results, name='encoder')
    #return decoded

# In[10]:

def simple_autoencoder(input_img):

    encoded = Dense(2304, activation='relu')(input_img)
    decoded = Dense(2304, activation='sigmoid')(encoded)
    
    return decoded

#num_classes = 1 
##flatten_shape = 2304,
##time_phase_input = Input(shape=(2304,))
#time_phase_input = Input(shape=(784,))
##time_phase_input = Input(shape=(flatten_shape))
#decoded_model = simple_autoencoder(time_phase_input)
#autoencoder = Model(time_phase_input, decoded_model)
#opt = optimizers.Adam()
#autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
#print(autoencoder.summary())

encoding_dim = 256 # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
#input_img = Input(shape=(2304,))
#input_img = Input(shape=(2304,))
input_img = Input(shape=input_shape)
autoencoder, encoder = create_convolutional_autoencoder(input_img)


print(input_img)
#
#
#encoded = Dense(encoding_dim, activation='relu')(input_img)
#encoded = Dense(128, activation='relu')(encoded)
##encoded = Dense(64, activation='relu')(encoded)
###encoded = Dropout(0.2)(encoded)
##encoded = Dense(32, activation='relu')(encoded)
##
##encoded = Dense(64, activation='relu')(encoded)
##encoded = Dense(128, activation='relu')(encoded)
#encoded = Dense(256, activation='relu')(encoded)
##encoded = Dropout(0.5)(encoded)
#decoded = Dense(2304, activation='softmax')(encoded)
#
#
#
#
## "encoded" is the encoded representation of the input
#
#
#
##encoded = Dense(encoding_dim, activation='relu', activity_regularizer=regularizers.l1(10e-5))(input_img)
## "decoded" is the lossy reconstruction of the input
##decoded = Dense(2304, activation='sigmoid')(encoded)
#autoencoder = Model(input_img, decoded)
#encoder = Model(input_img, decoded)
#
#
#encoder = Model(input_img, encoded)
## create a placeholder for an encoded (32-dimensional) input
#encoded_input = Input(shape=(256,))
## retrieve the last layer of the autoencoder model
#decoder_layer = autoencoder.layers[-1]
## create the decoder model
#decoder = Model(encoded_input, decoder_layer(encoded_input))
#opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
#
#Eautoencoder.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
print(autoencoder.summary())
# this model maps an input to its reconstruction
#autoencoder = Model(input_img, decoded)
# ## Do Train Test Split Correctly

# In[11]:


label_pulsars = np.ones(pulse_profile_pulsars_combined_array.shape[0], dtype=int)
label_nonpulsars = np.zeros(pulse_profile_nonpulsars_combined_array.shape[0], dtype=int)

## In[12]:
#
#
#freq_phase_data_combined = np.concatenate((reshaped_freq_phase_pulsars, reshaped_freq_phase_nonpulsars), axis = 0)
#freq_phase_label_combined = np.concatenate((label_pulsars, label_nonpulsars), axis = 0)
#freq_phase_train, freq_phase_test, freq_phase_label_train, freq_phase_label_test = train_test_split(freq_phase_data_combined, freq_phase_label_combined,                                          test_size=0.3, random_state=42)
#
#time_phase_data_combined = np.concatenate((reshaped_time_phase_pulsars, reshaped_time_phase_nonpulsars), axis = 0)
#time_phase_label_combined = np.concatenate((label_pulsars, label_nonpulsars), axis = 0)
#time_phase_train, time_phase_test, time_phase_label_train, time_phase_label_test = train_test_split(time_phase_data_combined, time_phase_label_combined,                                          test_size=0.3, random_state=42)
#
#pulse_profile_data_combined = np.concatenate((pulse_profile_pulsars_combined_array, pulse_profile_nonpulsars_combined_array), axis = 0)
#pulse_profile_label_combined = np.concatenate((label_pulsars, label_nonpulsars), axis = 0)
#pulse_profile_train, pulse_profile_test, pulse_profile_label_train, pulse_profile_label_test = train_test_split(pulse_profile_data_combined, pulse_profile_label_combined,                                          test_size=0.3, random_state=42)
#
#
#dm_curve_data_combined = np.concatenate((dm_curve_pulsars_combined_array, dm_curve_nonpulsars_combined_array), axis = 0)
#dm_curve_label_combined = np.concatenate((label_pulsars, label_nonpulsars), axis = 0)
#dm_curve_train, dm_curve_test, dm_curve_label_train, dm_curve_label_test = train_test_split(dm_curve_data_combined, dm_curve_label_combined,                                          test_size=0.3, random_state=42)
#es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
#mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)


## In[13]:
#
#
#model.fit([time_phase_train, freq_phase_train, pulse_profile_train, dm_curve_train], freq_phase_label_train , epochs=10, validation_split=0.3)
#model.fit([time_phase_train, pulse_profile_train, dm_curve_train], freq_phase_label_train , epochs=10, validation_split=0.3, callbacks=[es, mc])
#model.fit([time_phase_train, dm_curve_train], freq_phase_label_train , epochs=10, validation_split=0.3, callbacks=[es, mc])
#model.fit([time_phase_train], freq_phase_label_train , epochs=10, validation_split=0.3, callbacks=[es, mc])
label_pulsars_palfa = np.ones(time_phase_pulsars_palfa_data.shape[0], dtype=int)
label_nonpulsars_palfa = np.zeros(time_phase_nonpulsars_palfa_data.shape[0], dtype=int)

# Palfa + HTRU Test

freq_phase_data_combined = np.concatenate((reshaped_freq_phase_pulsars, reshaped_freq_phase_nonpulsars, reshaped_freq_phase_pulsars_palfa, reshaped_freq_phase_nonpulsars_palfa), axis = 0)
freq_phase_label_combined = np.concatenate((label_pulsars, label_nonpulsars, label_pulsars_palfa, label_nonpulsars_palfa), axis = 0)
freq_phase_train, freq_phase_test, freq_phase_label_train, freq_phase_label_test = train_test_split(freq_phase_data_combined, freq_phase_label_combined,                                          test_size=0.3, random_state=42)

time_phase_data_combined = np.concatenate((reshaped_time_phase_pulsars, reshaped_time_phase_nonpulsars, reshaped_time_phase_pulsars_palfa, reshaped_time_phase_nonpulsars_palfa), axis = 0)
time_phase_label_combined = np.concatenate((label_pulsars, label_nonpulsars, label_pulsars_palfa, label_nonpulsars_palfa), axis = 0)

#time_phase_data_combined = [np.reshape(f,(2304)) for f in time_phase_data_combined]

time_phase_train, time_phase_test, time_phase_label_train, time_phase_label_test = train_test_split(time_phase_data_combined, time_phase_label_combined,                                          test_size=0.3, random_state=42)

pulse_profile_data_combined = np.concatenate((pulse_profile_pulsars_combined_array, pulse_profile_nonpulsars_combined_array,pulse_profile_pulsars_palfa_data, pulse_profile_nonpulsars_palfa_data), axis = 0)
pulse_profile_label_combined = np.concatenate((label_pulsars, label_nonpulsars, label_pulsars_palfa, label_nonpulsars_palfa), axis = 0)
pulse_profile_train, pulse_profile_test, pulse_profile_label_train, pulse_profile_label_test = train_test_split(pulse_profile_data_combined, pulse_profile_label_combined,                                          test_size=0.3, random_state=42)


dm_curve_data_combined = np.concatenate((dm_curve_pulsars_combined_array, dm_curve_nonpulsars_combined_array, dm_curve_pulsars_palfa_data, dm_curve_nonpulsars_palfa_data), axis = 0)
dm_curve_label_combined = np.concatenate((label_pulsars, label_nonpulsars, label_pulsars_palfa, label_nonpulsars_palfa), axis = 0)
dm_curve_train, dm_curve_test, dm_curve_label_train, dm_curve_label_test = train_test_split(dm_curve_data_combined, dm_curve_label_combined,                                          test_size=0.3, random_state=42)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
mc = ModelCheckpoint('best_model_time_phase_only.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)


## In[13]:
#
#
#model.fit([time_phase_train, freq_phase_train, pulse_profile_train, dm_curve_train], freq_phase_label_train , epochs=20, validation_split=0.3)
#model.fit([time_phase_train], time_phase_label_train , epochs=20, validation_split=0.3)
time_phase_train = np.array(time_phase_train)
time_phase_test = np.array(time_phase_test)
#one_image = time_phase_train[0:9]
#one_label = time_phase_label_train[0:9]
##one_image = np.reshape(one_image, (48,48))
#
#plot_images(one_image, one_label)
#plt.imshow(one_image)

#plt.savefig('time_phase_example.png')
#print(np.shape(time_phase_train))
from keras.datasets import mnist
import numpy as np
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)

#autoencoder.fit(x_train, x_train,
#                epochs=50,
#                batch_size=256,
#                shuffle=True,
#                validation_data=(x_test, x_test))
#autoencoder.fit(time_phase_train, time_phase_train,  epochs=40, batch_size = 2048, shuffle=True, validation_split=0.3)
#autoencoder.save_weights('auto_encoder_time_phase.h5')
autoencoder.load_weights('auto_encoder_time_phase.h5')
### encode and decode some digits
### note that we take them from the *test* set
encoded_imgs = encoder.predict(time_phase_test)
decoded_imgs = autoencoder.predict(time_phase_test)
##print(type(encoded_imgs))
import matplotlib.pyplot as plt

n = 15  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(time_phase_test[i].reshape(48, 48))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    #ax = plt.subplot(2, n, i + 1 + n)
    #test = np.append(encoded_imgs[i], 0.)
    #plt.imshow(test.reshape(17, 17))
    #plt.gray()
    #ax.get_xaxis().set_visible(False)
    #ax.get_yaxis().set_visible(False)
    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(48, 48))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.savefig('autoencoder_results_new.png')
plt.show()

#model.load_weights('first_try.h5')


#print(encoder.output)
##encoder_flatten  = Flatten()(encoder.output)
##print(encoder_flatten)
print(type(encoder.output))

n_clusters=15
clustering_layer = ClusteringLayer(n_clusters, name='clustering')(encoder.output)
model = Model(inputs=encoder.input, outputs=[clustering_layer])
# Initialize cluster centers using k-means.
kmeans = KMeans(n_clusters=n_clusters, n_init=20)
y_pred = kmeans.fit_predict(encoder.predict(time_phase_train))
model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])
y_pred_last = np.copy(y_pred)
print(y_pred)
#acc = np.round(accuracy_score(time_phase_label_train, y_pred), 5)
#recall = np.round(recall_score(time_phase_label_train, y_pred), 5)
#precision = np.round(precision_score(time_phase_label_train, y_pred), 5)
#fscore = np.round(f1_score(time_phase_label_train, y_pred), 5)
#print('Acc = %.5f, recall = %.5f, precision = %.5f, fscore = %.5f' % (acc, recall, precision, fscore))

def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T

model.compile(optimizer=optimizers.SGD(0.01, 0.9), loss='kld')
loss = 0
index = 0
maxiter = 10000
update_interval = 120
index_array = np.arange(time_phase_train.shape[0])
tol = 0.0001
#print(index_array)
for ite in range(int(maxiter)):
    if ite % update_interval == 0:
        q = model.predict(time_phase_train, verbose=1)
        
        #print(np.shape(q))
        p = target_distribution(q)  # update the auxiliary target distribution p
        # evaluate the clustering performance
        y_pred = q.argmax(1)
        if time_phase_label_test is not None:
            acc = np.round(accuracy_score(time_phase_label_train, y_pred), 5)


        # check stop criterion
        delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
        y_pred_last = np.copy(y_pred)
        if ite > 0 and delta_label < tol:
            print('delta_label ', delta_label, '< tol ', tol)
            print('Reached tolerance threshold. Stopping training.')
            break
    idx = index_array[index * batch_size: min((index+1) * batch_size, time_phase_train.shape[0])]
    loss = model.train_on_batch(x=time_phase_train[idx], y=p[idx])
    index = index + 1 if (index + 1) * batch_size <= time_phase_train.shape[0] else 0
model.save_weights('clustering_model_results_time_phase.h5')
#model.load_weights('clustering_model_results_time_phase.h5')
q = model.predict(time_phase_train, verbose=1)
p = target_distribution(q)  # update the auxiliary target distribution p

# evaluate the clustering performance
y_pred = q.argmax(1)
print(q)
print(len(y_pred), len(time_phase_label_train))
#y_pred = np.asarray(y_pred)
df = pd.DataFrame({'File Name':time_phase_label_train, 'Score': y_pred})
print(df)
df.to_csv('clustering_results_8.csv', index=False)
df1 = df.loc[df['Score'] == 10]
df2 = df.loc[df['Score'] == 2]


n = 15  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(time_phase_test[i].reshape(48, 48))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    #ax = plt.subplot(2, n, i + 1 + n)
    #test = np.append(encoded_imgs[i], 0.)
    #plt.imshow(test.reshape(17, 17))
    #plt.gray()
    #ax.get_xaxis().set_visible(False)
    #ax.get_yaxis().set_visible(False)
    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(48, 48))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.savefig('autoencoder_results_new.png')
plt.show()
#with open('clustering_results_8.csv', 'w') as f:
#    for i in range(len(y_pred)):
#        f.write(time_phase_label_train[i] + ',' + y_pred[i] + '\n')
#        print(time_phase_label_train[i] + ',' + y_pred[i] + '\n')

#if time_phase_label_train is not None:
#    acc = np.round(accuracy_score(time_phase_label_train, y_pred), 5)
#    recall = np.round(recall_score(time_phase_label_train, y_pred), 5)
#    precision = np.round(precision_score(time_phase_label_train, y_pred), 5)
#    fscore = np.round(f1_score(time_phase_label_train, y_pred), 5)
#    loss = np.round(loss, 5)
#    print('Acc = %.5f, recall = %.5f, precision = %.5f, fscore = %.5f' % (acc, recall, precision, fscore), ' ; loss=', loss)
# In[14]:


#predict = model.predict([time_phase_test, freq_phase_test, pulse_profile_test, dm_curve_test])
#predict = model.predict([time_phase_test, pulse_profile_test, dm_curve_test])
#predict = model.predict([time_phase_test, dm_curve_test])
#predict = model.predict([time_phase_test])
#np.save('predictions.npy', predict)
#classified_results = np.rint(predict)
#classified_results = np.reshape(classified_results, len(classified_results))
##
##
### In[15]:
##
##
#
#f_score = f1_score(time_phase_label_test, classified_results, average='binary')
#precision = precision_score(time_phase_label_test, classified_results, average='binary')
#recall = recall_score(time_phase_label_test, classified_results, average='binary')
#accuracy = (time_phase_label_test == classified_results).sum()/len(time_phase_label_test)
#tn, fp, fn, tp = confusion_matrix(time_phase_label_test, classified_results).ravel()
#specificity = tn/(tn + fp)
#gmean = math.sqrt(specificity * recall)
#fpr = fp/(tn + fp)
#print('Results with HTRU-S Lowlat Survey and PALFA Combined:')
#print('Accuracy:', accuracy, 'F Score:', f_score, 'Precision:', precision, 'Recall:', recall, 'False Positive Rate:', fpr, 'Specificity:', specificity, 'G-Mean:', gmean)

#f_score = f1_score(time_phase_label_test, classified_results, average='binary')
#precision = precision_score(time_phase_label_test, classified_results, average='binary')
#recall = recall_score(time_phase_label_test, classified_results, average='binary')
#print('Results with HTRU-S Lowlat Survey:')
#print('F Score:', f_score, 'Precision:', precision, 'Recall:', recall)

#Palfa Test
#print('Palfa Data Test Begins!')
#
#
#label_pulsars_palfa = np.ones(time_phase_pulsars_palfa_data.shape[0], dtype=int)
#label_nonpulsars_palfa = np.zeros(time_phase_nonpulsars_palfa_data.shape[0], dtype=int)
#
#freq_phase_data_combined_palfa = np.concatenate((reshaped_freq_phase_pulsars_palfa, reshaped_freq_phase_nonpulsars_palfa), axis = 0)
#freq_phase_label_combined_palfa = np.concatenate((label_pulsars_palfa, label_nonpulsars_palfa), axis = 0)
#
#time_phase_data_combined_palfa = np.concatenate((reshaped_time_phase_pulsars_palfa, reshaped_time_phase_nonpulsars_palfa), axis = 0)
#time_phase_label_combined_palfa = np.concatenate((label_pulsars_palfa, label_nonpulsars_palfa), axis = 0)
#
#pulse_profile_data_combined_palfa = np.concatenate((pulse_profile_pulsars_palfa_data, pulse_profile_nonpulsars_palfa_data), axis = 0)
#pulse_profile_label_combined_palfa = np.concatenate((label_pulsars_palfa, label_nonpulsars_palfa), axis = 0)
#
#
#dm_curve_data_combined_palfa = np.concatenate((dm_curve_pulsars_palfa_data, dm_curve_nonpulsars_palfa_data), axis = 0)
#dm_curve_label_combined_palfa = np.concatenate((label_pulsars_palfa, label_nonpulsars_palfa), axis = 0)
#
#
##predict = model.predict([time_phase_data_combined_palfa, freq_phase_data_combined_palfa, pulse_profile_data_combined_palfa, dm_curve_data_combined_palfa])
##predict = model.predict([time_phase_data_combined_palfa, pulse_profile_data_combined_palfa, dm_curve_data_combined_palfa])
##predict = model.predict([time_phase_data_combined_palfa, dm_curve_data_combined_palfa])
#predict = model.predict([time_phase_data_combined_palfa])
#np.save('predictions_palfa.npy', predict)
#classified_results = np.rint(predict)
#classified_results = np.reshape(classified_results, len(classified_results))
#
#
## In[26]:
#
#
#f_score = f1_score(time_phase_label_combined_palfa, classified_results, average='binary')
#precision = precision_score(time_phase_label_combined_palfa, classified_results, average='binary')
#recall = recall_score(time_phase_label_combined_palfa, classified_results, average='binary')
#accuracy = (time_phase_label_combined_palfa == classified_results).sum()/len(time_phase_label_combined_palfa)
#print('Palfa Final Results:')
#print('F Score:', f_score, 'Precision:', precision, 'Recall:', recall, 'Accuracy:', accuracy)



#accuracy = (time_phase_label_test == classified_results).sum()/len(time_phase_label_test)
#accuracy = (pulse_profile_label_test == classified_results).sum()/len(pulse_profile_label_test)
#print('F Score:', f_score, 'Precision:', precision, 'Recall:', recall, 'Accuracy:', accuracy)
#np.shape(classified_results[:,1])

#
## ## GBNCC Test
#
## In[16]:
#
#
#pulse_profile_gbncc_pulsars = np.load('input_data/pulse_profile_gbncc_test_data_pulsars.npy')
#pulse_profile_gbncc_nonpulsars = np.load('input_data/pulse_profile_gbncc_test_data_nonpulsars_part1.npy')
#time_phase_gbncc_pulsars = np.load('input_data/time_phase_gbncc_test_data_pulsars.npy')
#
#
## In[17]:
#
#
#classified_results = classified_results.astype(int)
#
#
## In[18]:
#
#
#np.shape(pulse_profile_gbncc_pulsars), np.shape(time_phase_gbncc_pulsars)
#
#
## In[19]:
#
#
#(time_phase_label_test == classified_results).sum()
#
#
## In[20]:
#
#
#(pulse_profile_label_test == classified_results).sum()
#
#
## In[21]:
#
#
#pulse_profile_label_test.shape, classified_results.shape
#
#
## # PALFA TEST
#
## In[22]:
#
#
## Load all data
#time_phase_pulsars = np.load('palfa_data/time_phase_data_palfa_data_pulsars.npy')
#time_phase_nonpulsars = np.load('palfa_data/time_phase_data_palfa_data_nonpulsars.npy')
#
#freq_phase_pulsars = np.load('palfa_data/freq_phase_data_palfa_data_pulsars.npy')
#freq_phase_nonpulsars = np.load('palfa_data/freq_phase_data_palfa_data_nonpulsars.npy')
#
#pulse_profile_pulsars = np.load('palfa_data/pulse_profile_data_palfa_data_pulsars.npy')
#pulse_profile_nonpulsars = np.load('palfa_data/pulse_profile_data_palfa_data_nonpulsars.npy')
#
#dm_curve_pulsars = np.load('palfa_data/dm_curve_data_palfa_data_pulsars.npy')
#dm_curve_nonpulsars = np.load('palfa_data/dm_curve_data_palfa_data_nonpulsars.npy')
#
##pulse_profile_gbncc_pulsars = np.load('input_data/pulse_profile_gbncc_test_data_pulsars.npy')
##pulse_profile_gbncc_nonpulsars = np.load('input_data/pulse_profile_gbncc_test_data_nonpulsars_part1.npy')
#
#reshaped_time_phase_pulsars = [np.reshape(f,(48,48,1)) for f in time_phase_pulsars]
#reshaped_time_phase_nonpulsars = [np.reshape(f,(48,48,1)) for f in time_phase_nonpulsars]
#
#reshaped_freq_phase_pulsars = [np.reshape(f,(48,48,1)) for f in freq_phase_pulsars]
#reshaped_freq_phase_nonpulsars = [np.reshape(f,(48,48,1)) for f in freq_phase_nonpulsars]
#
#print('Total Number of Pulsar Examples is %d' %len(dm_curve_pulsars))
#print('Total Number of Non-Pulsar Examples is %d' %len(dm_curve_nonpulsars))
#
#
## In[23]:
#
#
#label_pulsars = np.ones(len(dm_curve_pulsars), dtype=int)
#label_nonpulsars = np.zeros(len(dm_curve_nonpulsars), dtype=int)
#
#
## In[24]:
#
#
#freq_phase_data_combined = np.concatenate((reshaped_freq_phase_pulsars, reshaped_freq_phase_nonpulsars), axis = 0)
#freq_phase_label_combined = np.concatenate((label_pulsars, label_nonpulsars), axis = 0)
#
#time_phase_data_combined = np.concatenate((reshaped_time_phase_pulsars, reshaped_time_phase_nonpulsars), axis = 0)
#time_phase_label_combined = np.concatenate((label_pulsars, label_nonpulsars), axis = 0)
#
#pulse_profile_data_combined = np.concatenate((pulse_profile_pulsars, pulse_profile_nonpulsars), axis = 0)
#pulse_profile_label_combined = np.concatenate((label_pulsars, label_nonpulsars), axis = 0)
#
#dm_curve_data_combined = np.concatenate((dm_curve_pulsars, dm_curve_nonpulsars), axis = 0)
#dm_curve_label_combined = np.concatenate((label_pulsars, label_nonpulsars), axis = 0)
#
#
## In[25]:
#
#
#predict = model.predict([time_phase_data_combined, freq_phase_data_combined, pulse_profile_data_combined, dm_curve_data_combined])
#np.save('predictions.npy', predict)
#classified_results = np.rint(predict)
#classified_results = np.reshape(classified_results, len(classified_results))
#
#
## In[26]:
#
#
#f_score = f1_score(time_phase_label_combined, classified_results, average='binary')
#precision = precision_score(time_phase_label_combined, classified_results, average='binary')
#recall = recall_score(time_phase_label_combined, classified_results, average='binary')
#accuracy = (time_phase_label_combined == classified_results).sum()/len(time_phase_label_combined)
#print('F Score:', f_score, 'Precision:', precision, 'Recall:', recall, 'Accuracy:', accuracy)
#
#
## In[28]:
#
#
#class_names = ['Non-Pulsar','Pulsar']
#
#cnf_matrix = confusion_matrix(time_phase_label_combined, classified_results)
#plt.figure()
#plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix, without normalization')
#plt.figure()
#plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title='Confusion matrix, with normalization')
#plt.show()
#
#
## # Batch 2 HTRU Test, Tricky Pulsars
#
## In[12]:
#
#
## Load all data
#time_phase_pulsars = np.load('lowlat_cands/pulsars/time_phase_data_new_batch_pulsars_batch_1.npy')
#time_phase_nonpulsars = np.load('lowlat_cands/pulsars/time_phase_data_nonpulsars.npy')
#
#freq_phase_pulsars = np.load('lowlat_cands/pulsars/freq_phase_data_new_batch_pulsars_batch_1.npy')
#freq_phase_nonpulsars = np.load('lowlat_cands/pulsars/freq_phase_data_nonpulsars.npy')
#
#pulse_profile_pulsars = np.load('lowlat_cands/pulsars/pulse_profile_data_new_batch_pulsars_batch_1.npy')
#pulse_profile_nonpulsars = np.load('lowlat_cands/pulsars/pulse_profile_data_nonpulsars.npy')
#
#dm_curve_pulsars = np.load('lowlat_cands/pulsars/dm_curve_data_new_batch_pulsars_batch_1.npy')
#dm_curve_nonpulsars = np.load('lowlat_cands/pulsars/dm_curve_data_nonpulsars.npy')
#
##pulse_profile_gbncc_pulsars = np.load('input_data/pulse_profile_gbncc_test_data_pulsars.npy')
##pulse_profile_gbncc_nonpulsars = np.load('input_data/pulse_profile_gbncc_test_data_nonpulsars_part1.npy')
#
#reshaped_time_phase_pulsars = [np.reshape(f,(48,48,1)) for f in time_phase_pulsars]
#reshaped_time_phase_nonpulsars = [np.reshape(f,(48,48,1)) for f in time_phase_nonpulsars]
#
#reshaped_freq_phase_pulsars = [np.reshape(f,(48,48,1)) for f in freq_phase_pulsars]
#reshaped_freq_phase_nonpulsars = [np.reshape(f,(48,48,1)) for f in freq_phase_nonpulsars]
#
#print('Total Number of Pulsar Examples is %d' %len(dm_curve_pulsars))
#print('Total Number of Non-Pulsar Examples is %d' %len(dm_curve_nonpulsars))
#
#label_pulsars = np.ones(len(dm_curve_pulsars), dtype=int)
#label_nonpulsars = np.zeros(len(dm_curve_nonpulsars), dtype=int)
#
#freq_phase_data_combined = np.concatenate((reshaped_freq_phase_pulsars, reshaped_freq_phase_nonpulsars), axis = 0)
#freq_phase_label_combined = np.concatenate((label_pulsars, label_nonpulsars), axis = 0)
#
#time_phase_data_combined = np.concatenate((reshaped_time_phase_pulsars, reshaped_time_phase_nonpulsars), axis = 0)
#time_phase_label_combined = np.concatenate((label_pulsars, label_nonpulsars), axis = 0)
#
#pulse_profile_data_combined = np.concatenate((pulse_profile_pulsars, pulse_profile_nonpulsars), axis = 0)
#pulse_profile_label_combined = np.concatenate((label_pulsars, label_nonpulsars), axis = 0)
#
#dm_curve_data_combined = np.concatenate((dm_curve_pulsars, dm_curve_nonpulsars), axis = 0)
#dm_curve_label_combined = np.concatenate((label_pulsars, label_nonpulsars), axis = 0)
#model.load_weights('first_try.h5')
#predict = model.predict([time_phase_data_combined, freq_phase_data_combined, pulse_profile_data_combined, dm_curve_data_combined])
#np.save('predictions.npy', predict)
#classified_results = np.rint(predict)
#classified_results = np.reshape(classified_results, len(classified_results))
#
#f_score = f1_score(time_phase_label_combined, classified_results, average='binary')
#precision = precision_score(time_phase_label_combined, classified_results, average='binary')
#recall = recall_score(time_phase_label_combined, classified_results, average='binary')
#accuracy = (time_phase_label_combined == classified_results).sum()/len(time_phase_label_combined)
#print('F Score:', f_score, 'Precision:', precision, 'Recall:', recall, 'Accuracy:', accuracy)
#
#class_names = ['Non-Pulsar','Pulsar']
#
#cnf_matrix = confusion_matrix(time_phase_label_combined, classified_results)
#plt.figure()
#plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix, without normalization')
#plt.figure()
#plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title='Confusion matrix, with normalization')
#plt.show()
#
#
## In[ ]:
#
#
#
#
