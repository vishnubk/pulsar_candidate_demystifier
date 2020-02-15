#!/usr/bin/env python
# coding: utf-8



import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Conv3D, MaxPooling3D
from keras.layers import Activation, Dropout, Flatten, Dense, LeakyReLU
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from keras import backend as K
from keras import optimizers
import numpy as np
import math, time
import itertools
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, concatenate
from keras.models import Model
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from my_classes import DataGenerator
import glob
import pandas as pd
from keras.backend.tensorflow_backend import set_session
from keras.backend.tensorflow_backend import clear_session
from keras.backend.tensorflow_backend import get_session
import gc
# Reset Keras Session
def reset_keras():
    sess = get_session()
    clear_session()
    sess.close()
    sess = get_session()

    try:
        del classifier # this is from global space - change this as you need
    except:
        pass

    print(gc.collect()) # if it's done something you should see a number being outputted

    # use the same config as you used to create the session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.visible_device_list = "0"
    set_session(tf.Session(config=config))

reset_keras()
#df = pd.read_csv('lowlat_cands_full_labelled.csv', header=None, names = ['filename', 'label'])
df = pd.read_csv('reshaped_lowlat_cands_full_labelled.csv', header=None, names = ['filename', 'label'])

#pulsars_files = sorted(glob.glob('data/full_3D_pfd_data_pulsars_batch_*.npy')) 
#nonpulsars_files = sorted(glob.glob('data/full_3D_pfd_data_nonpulsars_batch_*.npy')) 

params = {'dim': (64,64,64),
          'batch_size': 60,
          'n_classes': 2,
          'n_channels': 1,
          'shuffle': True}
#pulsars_data = [np.load(f) for f in pulsars_files]
#nonpulsars_data = [np.load(f) for f in nonpulsars_files]
##shape = [np.concatenate((time_phase_pulsars_combined_array, f), axis = 0) for f in time_phase_pulsars_data]
#
#for f in pulsars_data:
#    pulsars_combined_array = np.concatenate((pulsars_combined_array, f), axis = 0)
#
#for f in nonpulsars_data:
#    nonpulsars_combined_array = np.concatenate((nonpulsars_combined_array, f), axis = 0)
#
#
#print('Total Number of Pulsar Examples is %d' %pulsars_combined_array.shape[0])
#print('Total Number of Non-Pulsar Examples is %d' %nonpulsars_combined_array.shape[0])
#print('Total Labelled Data Available is %d'%(pulsars_combined_array.shape[0] + nonpulsars_combined_array.shape[0]))
#
#
##Load Palfa Data
#
#time_phase_pulsars_palfa_data = np.load('palfa_data/time_phase_data_palfa_data_pulsars.npy')
#freq_phase_pulsars_palfa_data = np.load('palfa_data/freq_phase_data_palfa_data_pulsars.npy')
#pulse_profile_pulsars_palfa_data = np.load('palfa_data/pulse_profile_data_palfa_data_pulsars.npy')
#dm_curve_pulsars_palfa_data = np.load('palfa_data/dm_curve_data_palfa_data_pulsars.npy')
#
#time_phase_nonpulsars_palfa_data = np.load('palfa_data/time_phase_data_palfa_data_nonpulsars.npy')
#freq_phase_nonpulsars_palfa_data = np.load('palfa_data/freq_phase_data_palfa_data_nonpulsars.npy')
#pulse_profile_nonpulsars_palfa_data = np.load('palfa_data/pulse_profile_data_palfa_data_nonpulsars.npy')
#dm_curve_nonpulsars_palfa_data = np.load('palfa_data/dm_curve_data_palfa_data_nonpulsars.npy')
#
#
#print('Total Number of Palfa Pulsar Examples is %d' %dm_curve_nonpulsars_palfa_data.shape[0])
#print('Total Number of Palfa Non-Pulsar Examples is %d' %freq_phase_pulsars_palfa_data.shape[0])
##Reshape Palfa Data
#
#reshaped_time_phase_pulsars_palfa = [np.reshape(f,(48,48,1)) for f in time_phase_pulsars_palfa_data]
#reshaped_time_phase_nonpulsars_palfa = [np.reshape(f,(48,48,1)) for f in time_phase_nonpulsars_palfa_data]
#
#reshaped_freq_phase_pulsars_palfa = [np.reshape(f,(48,48,1)) for f in freq_phase_pulsars_palfa_data]
#reshaped_freq_phase_nonpulsars_palfa = [np.reshape(f,(48,48,1)) for f in freq_phase_nonpulsars_palfa_data]
#
##
##
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
    plt.show()


#reset_keras()
#
data_dimension_size = 64
#img_width, img_height = 48, 48
input_shape = (data_dimension_size, data_dimension_size, data_dimension_size,1)
##tensor_board = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
##NAME = "Simple-CNN"
##tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
if K.image_data_format() == 'channels_first':
    input_shape = (1, data_dimension_size, data_dimension_size, data_dimension_size)
else:
    input_shape = (data_dimension_size, data_dimension_size, data_dimension_size, 1)

# Generator is used to avoid reading entire data into system memory.
#def generator(batch_size,from_list_x,from_list_y):
#
#    assert len(from_list_x) == len(from_list_y)
#    total_size = len(from_list_x)
#
#    while True:
#
#        for i in range(0,total_size,batch_size):
#            yield np.array(from_list_x[i:i+batch_size]), np.array(from_list_y[i:i+batch_size])


## In[9]:
#
#
def create_2dconvolution_layers(input_img):
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

def create_3D_conv_net(input_data):

    model = Sequential()
    model.add(Conv3D(32, kernel_size=(3, 3, 3), padding='same', activation='relu', input_shape=input_data))
    #model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    #model.add(Dropout(0.25))
    model.add(Conv3D(32, kernel_size=(3, 3, 3), padding='same', activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Conv3D(64, kernel_size=(3, 3, 3), padding='same', activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(8, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    opt = optimizers.Adam()

    model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

    return model
    
# In[10]:

num_classes = 2
data_input = Input(shape=input_shape)
model = create_3D_conv_net(input_shape)




#
#
 ## Do Train Test Split Correctly
#68 % training, 12 % valdidation and 20 % test
train, test = train_test_split(df, test_size=0.2, random_state=42)

#labels_train_and_validation = train.set_index('filename').to_dict()
labels_train_and_validation = dict(zip(train['filename'], train['label']))

#train, validation = train_test_split(train, test_size=0.15, random_state=42)
#print(test)
train_validation_list = train['filename'].tolist()
train, validation = train_test_split(train_validation_list, test_size=0.15, random_state=12)
keys = ['train', 'validation']
partition = dict.fromkeys(keys)
partition['train'] = train
partition['validation'] = validation
training_generator = DataGenerator(partition['train'], labels_train_and_validation, **params)
validation_generator = DataGenerator(partition['validation'], labels_train_and_validation, **params)
#print(type(partition['train']), type(partition['validation']))
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
mc = ModelCheckpoint('best_model_3d_convnet.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
print(model.summary())
model.fit_generator(generator=training_generator, validation_data=validation_generator, epochs = 10, use_multiprocessing=True,
                    workers=4)


#
#
#
### In[13]:
##
##
#model.fit([time_phase_train, freq_phase_train, pulse_profile_train, dm_curve_train], freq_phase_label_train , epochs=10, validation_split=0.3)
#
#
#
model.save_weights('3D_Convnet_first_try.h5')
#model.load_weights('first_try.h5')

#
#
#
#
#
#
#
## In[14]:
#
#
#predict = model.predict([time_phase_test, freq_phase_test, pulse_profile_test, dm_curve_test])
##predict = model.predict([time_phase_test, pulse_profile_test, dm_curve_test])
##predict = model.predict([time_phase_test, dm_curve_test])
##predict = model.predict([time_phase_test])
#np.save('predictions.npy', predict)
#classified_results = np.rint(predict)
#classified_results = np.reshape(classified_results, len(classified_results))
##
##
### In[15]:
##
##
#f_score = f1_score(time_phase_label_test, classified_results, average='binary')
#precision = precision_score(time_phase_label_test, classified_results, average='binary')
#recall = recall_score(time_phase_label_test, classified_results, average='binary')
#print('Results with HTRU-S Lowlat Survey:')
#print('F Score:', f_score, 'Precision:', precision, 'Recall:', recall)
#
##Palfa Test
##print('Palfa Data Test Begins!')
##
##
##label_pulsars_palfa = np.ones(time_phase_pulsars_palfa_data.shape[0], dtype=int)
##label_nonpulsars_palfa = np.zeros(time_phase_nonpulsars_palfa_data.shape[0], dtype=int)
##
##freq_phase_data_combined_palfa = np.concatenate((reshaped_freq_phase_pulsars_palfa, reshaped_freq_phase_nonpulsars_palfa), axis = 0)
##freq_phase_label_combined_palfa = np.concatenate((label_pulsars_palfa, label_nonpulsars_palfa), axis = 0)
##
##time_phase_data_combined_palfa = np.concatenate((reshaped_time_phase_pulsars_palfa, reshaped_time_phase_nonpulsars_palfa), axis = 0)
##time_phase_label_combined_palfa = np.concatenate((label_pulsars_palfa, label_nonpulsars_palfa), axis = 0)
##
##pulse_profile_data_combined_palfa = np.concatenate((pulse_profile_pulsars_palfa_data, pulse_profile_nonpulsars_palfa_data), axis = 0)
##pulse_profile_label_combined_palfa = np.concatenate((label_pulsars_palfa, label_nonpulsars_palfa), axis = 0)
##
##
##dm_curve_data_combined_palfa = np.concatenate((dm_curve_pulsars_palfa_data, dm_curve_nonpulsars_palfa_data), axis = 0)
##dm_curve_label_combined_palfa = np.concatenate((label_pulsars_palfa, label_nonpulsars_palfa), axis = 0)
##
##
###predict = model.predict([time_phase_data_combined_palfa, freq_phase_data_combined_palfa, pulse_profile_data_combined_palfa, dm_curve_data_combined_palfa])
###predict = model.predict([time_phase_data_combined_palfa, pulse_profile_data_combined_palfa, dm_curve_data_combined_palfa])
###predict = model.predict([time_phase_data_combined_palfa, dm_curve_data_combined_palfa])
##predict = model.predict([time_phase_data_combined_palfa])
##np.save('predictions_palfa.npy', predict)
##classified_results = np.rint(predict)
##classified_results = np.reshape(classified_results, len(classified_results))
##
##
### In[26]:
##
##
##f_score = f1_score(time_phase_label_combined_palfa, classified_results, average='binary')
##precision = precision_score(time_phase_label_combined_palfa, classified_results, average='binary')
##recall = recall_score(time_phase_label_combined_palfa, classified_results, average='binary')
##accuracy = (time_phase_label_combined_palfa == classified_results).sum()/len(time_phase_label_combined_palfa)
##print('Palfa Final Results:')
##print('F Score:', f_score, 'Precision:', precision, 'Recall:', recall, 'Accuracy:', accuracy)
#
#
#
##accuracy = (time_phase_label_test == classified_results).sum()/len(time_phase_label_test)
##accuracy = (pulse_profile_label_test == classified_results).sum()/len(pulse_profile_label_test)
##print('F Score:', f_score, 'Precision:', precision, 'Recall:', recall, 'Accuracy:', accuracy)
##np.shape(classified_results[:,1])
#
##
### ## GBNCC Test
##
### In[16]:
##
##
##pulse_profile_gbncc_pulsars = np.load('input_data/pulse_profile_gbncc_test_data_pulsars.npy')
##pulse_profile_gbncc_nonpulsars = np.load('input_data/pulse_profile_gbncc_test_data_nonpulsars_part1.npy')
##time_phase_gbncc_pulsars = np.load('input_data/time_phase_gbncc_test_data_pulsars.npy')
##
##
### In[17]:
##
##
##classified_results = classified_results.astype(int)
##
##
### In[18]:
##
##
##np.shape(pulse_profile_gbncc_pulsars), np.shape(time_phase_gbncc_pulsars)
##
##
### In[19]:
##
##
##(time_phase_label_test == classified_results).sum()
##
##
### In[20]:
##
##
##(pulse_profile_label_test == classified_results).sum()
##
##
### In[21]:
##
##
##pulse_profile_label_test.shape, classified_results.shape
##
##
### # PALFA TEST
##
### In[22]:
##
##
### Load all data
##time_phase_pulsars = np.load('palfa_data/time_phase_data_palfa_data_pulsars.npy')
##time_phase_nonpulsars = np.load('palfa_data/time_phase_data_palfa_data_nonpulsars.npy')
##
##freq_phase_pulsars = np.load('palfa_data/freq_phase_data_palfa_data_pulsars.npy')
##freq_phase_nonpulsars = np.load('palfa_data/freq_phase_data_palfa_data_nonpulsars.npy')
##
##pulse_profile_pulsars = np.load('palfa_data/pulse_profile_data_palfa_data_pulsars.npy')
##pulse_profile_nonpulsars = np.load('palfa_data/pulse_profile_data_palfa_data_nonpulsars.npy')
##
##dm_curve_pulsars = np.load('palfa_data/dm_curve_data_palfa_data_pulsars.npy')
##dm_curve_nonpulsars = np.load('palfa_data/dm_curve_data_palfa_data_nonpulsars.npy')
##
###pulse_profile_gbncc_pulsars = np.load('input_data/pulse_profile_gbncc_test_data_pulsars.npy')
###pulse_profile_gbncc_nonpulsars = np.load('input_data/pulse_profile_gbncc_test_data_nonpulsars_part1.npy')
##
##reshaped_time_phase_pulsars = [np.reshape(f,(48,48,1)) for f in time_phase_pulsars]
##reshaped_time_phase_nonpulsars = [np.reshape(f,(48,48,1)) for f in time_phase_nonpulsars]
##
##reshaped_freq_phase_pulsars = [np.reshape(f,(48,48,1)) for f in freq_phase_pulsars]
##reshaped_freq_phase_nonpulsars = [np.reshape(f,(48,48,1)) for f in freq_phase_nonpulsars]
##
##print('Total Number of Pulsar Examples is %d' %len(dm_curve_pulsars))
##print('Total Number of Non-Pulsar Examples is %d' %len(dm_curve_nonpulsars))
##
##
### In[23]:
##
##
##label_pulsars = np.ones(len(dm_curve_pulsars), dtype=int)
##label_nonpulsars = np.zeros(len(dm_curve_nonpulsars), dtype=int)
##
##
### In[24]:
##
##
##freq_phase_data_combined = np.concatenate((reshaped_freq_phase_pulsars, reshaped_freq_phase_nonpulsars), axis = 0)
##freq_phase_label_combined = np.concatenate((label_pulsars, label_nonpulsars), axis = 0)
##
##time_phase_data_combined = np.concatenate((reshaped_time_phase_pulsars, reshaped_time_phase_nonpulsars), axis = 0)
##time_phase_label_combined = np.concatenate((label_pulsars, label_nonpulsars), axis = 0)
##
##pulse_profile_data_combined = np.concatenate((pulse_profile_pulsars, pulse_profile_nonpulsars), axis = 0)
##pulse_profile_label_combined = np.concatenate((label_pulsars, label_nonpulsars), axis = 0)
##
##dm_curve_data_combined = np.concatenate((dm_curve_pulsars, dm_curve_nonpulsars), axis = 0)
##dm_curve_label_combined = np.concatenate((label_pulsars, label_nonpulsars), axis = 0)
##
##
### In[25]:
##
##
##predict = model.predict([time_phase_data_combined, freq_phase_data_combined, pulse_profile_data_combined, dm_curve_data_combined])
##np.save('predictions.npy', predict)
##classified_results = np.rint(predict)
##classified_results = np.reshape(classified_results, len(classified_results))
##
##
### In[26]:
##
##
##f_score = f1_score(time_phase_label_combined, classified_results, average='binary')
##precision = precision_score(time_phase_label_combined, classified_results, average='binary')
##recall = recall_score(time_phase_label_combined, classified_results, average='binary')
##accuracy = (time_phase_label_combined == classified_results).sum()/len(time_phase_label_combined)
##print('F Score:', f_score, 'Precision:', precision, 'Recall:', recall, 'Accuracy:', accuracy)
##
##
### In[28]:
##
##
##class_names = ['Non-Pulsar','Pulsar']
##
##cnf_matrix = confusion_matrix(time_phase_label_combined, classified_results)
##plt.figure()
##plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix, without normalization')
##plt.figure()
##plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title='Confusion matrix, with normalization')
##plt.show()
##
##
### # Batch 2 HTRU Test, Tricky Pulsars
##
### In[12]:
##
##
### Load all data
##time_phase_pulsars = np.load('lowlat_cands/pulsars/time_phase_data_new_batch_pulsars_batch_1.npy')
##time_phase_nonpulsars = np.load('lowlat_cands/pulsars/time_phase_data_nonpulsars.npy')
##
##freq_phase_pulsars = np.load('lowlat_cands/pulsars/freq_phase_data_new_batch_pulsars_batch_1.npy')
##freq_phase_nonpulsars = np.load('lowlat_cands/pulsars/freq_phase_data_nonpulsars.npy')
##
##pulse_profile_pulsars = np.load('lowlat_cands/pulsars/pulse_profile_data_new_batch_pulsars_batch_1.npy')
##pulse_profile_nonpulsars = np.load('lowlat_cands/pulsars/pulse_profile_data_nonpulsars.npy')
##
##dm_curve_pulsars = np.load('lowlat_cands/pulsars/dm_curve_data_new_batch_pulsars_batch_1.npy')
##dm_curve_nonpulsars = np.load('lowlat_cands/pulsars/dm_curve_data_nonpulsars.npy')
##
###pulse_profile_gbncc_pulsars = np.load('input_data/pulse_profile_gbncc_test_data_pulsars.npy')
###pulse_profile_gbncc_nonpulsars = np.load('input_data/pulse_profile_gbncc_test_data_nonpulsars_part1.npy')
##
##reshaped_time_phase_pulsars = [np.reshape(f,(48,48,1)) for f in time_phase_pulsars]
##reshaped_time_phase_nonpulsars = [np.reshape(f,(48,48,1)) for f in time_phase_nonpulsars]
##
##reshaped_freq_phase_pulsars = [np.reshape(f,(48,48,1)) for f in freq_phase_pulsars]
##reshaped_freq_phase_nonpulsars = [np.reshape(f,(48,48,1)) for f in freq_phase_nonpulsars]
##
##print('Total Number of Pulsar Examples is %d' %len(dm_curve_pulsars))
##print('Total Number of Non-Pulsar Examples is %d' %len(dm_curve_nonpulsars))
##
##label_pulsars = np.ones(len(dm_curve_pulsars), dtype=int)
##label_nonpulsars = np.zeros(len(dm_curve_nonpulsars), dtype=int)
##
##freq_phase_data_combined = np.concatenate((reshaped_freq_phase_pulsars, reshaped_freq_phase_nonpulsars), axis = 0)
##freq_phase_label_combined = np.concatenate((label_pulsars, label_nonpulsars), axis = 0)
##
##time_phase_data_combined = np.concatenate((reshaped_time_phase_pulsars, reshaped_time_phase_nonpulsars), axis = 0)
##time_phase_label_combined = np.concatenate((label_pulsars, label_nonpulsars), axis = 0)
##
##pulse_profile_data_combined = np.concatenate((pulse_profile_pulsars, pulse_profile_nonpulsars), axis = 0)
##pulse_profile_label_combined = np.concatenate((label_pulsars, label_nonpulsars), axis = 0)
##
##dm_curve_data_combined = np.concatenate((dm_curve_pulsars, dm_curve_nonpulsars), axis = 0)
##dm_curve_label_combined = np.concatenate((label_pulsars, label_nonpulsars), axis = 0)
##model.load_weights('first_try.h5')
##predict = model.predict([time_phase_data_combined, freq_phase_data_combined, pulse_profile_data_combined, dm_curve_data_combined])
##np.save('predictions.npy', predict)
##classified_results = np.rint(predict)
##classified_results = np.reshape(classified_results, len(classified_results))
##
##f_score = f1_score(time_phase_label_combined, classified_results, average='binary')
##precision = precision_score(time_phase_label_combined, classified_results, average='binary')
##recall = recall_score(time_phase_label_combined, classified_results, average='binary')
##accuracy = (time_phase_label_combined == classified_results).sum()/len(time_phase_label_combined)
##print('F Score:', f_score, 'Precision:', precision, 'Recall:', recall, 'Accuracy:', accuracy)
##
##class_names = ['Non-Pulsar','Pulsar']
##
##cnf_matrix = confusion_matrix(time_phase_label_combined, classified_results)
##plt.figure()
##plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix, without normalization')
##plt.figure()
##plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title='Confusion matrix, with normalization')
##plt.show()
##
##
### In[ ]:
##
##
##
##
