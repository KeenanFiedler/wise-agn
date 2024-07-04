import numpy as np
import h5py as h5
import numpy as np 
import scipy.stats


from keras.models import Model, load_model
import keras.layers as layers
import keras
from keras import ops
from keras.layers import Dense, LeakyReLU, Lambda, Input, Conv1D, MaxPooling1D, AveragePooling1D, UpSampling1D, Reshape, Flatten,Cropping1D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import plot_model
from tensorflow import math as K
import tensorflow as tf
from sklearn.model_selection import train_test_split

original_dim = 119
latent_dim = 6
batch_size = 64
epochs = 10


infile = h5.File("C:\\Users\\keena\\Documents\\University of Arizona\\Jobs\\TIMESTEP NOIRLAB\\wise-agn\\pca\\clumpy_models_201410_tvavg.hdf5", 'r')
output_flux = np.log10(infile['flux_tor'][:])
Y = infile['Y'][:]
N0 = infile['N0'][:]
i = infile['i'][:]
q = infile['q'][:]
sig = infile['sig'][:]
tv = infile['tv'][:]

"""
infile = h5.File("C:\\Users\\keena\\Documents\\University of Arizona\\Jobs\\TIMESTEP NOIRLAB\\wise-agn\\pca\\models_subset.hdf5", 'r')
output_flux = np.log10(infile['dset']['flux_tor'][:])
wave = infile['dset']['wave'][:]
Y = infile['dset']['Y'][:]
N0 = infile['dset']['N0'][:]
i = infile['dset']['i'][:]
q = infile['dset']['q'][:]
sig = infile['dset']['sig'][:]
tv = infile['dset']['tv'][:]
"""

input_params = Y.reshape(len(Y),1)
input_params = np.append(input_params, N0.reshape(len(N0),1),1)
input_params = np.append(input_params, i.reshape(len(i),1),1)
input_params = np.append(input_params, q.reshape(len(q),1),1)
input_params = np.append(input_params, sig.reshape(len(sig),1),1)
input_params = np.append(input_params, tv.reshape(len(tv),1),1)


in_temp, in_test, out_temp, out_test = train_test_split(input_params, output_flux, test_size=0.05, random_state=1)
in_train, in_valid, out_train, out_valid = train_test_split(in_temp, out_temp, test_size=0.1, random_state=1)

checkpointer = ModelCheckpoint('model_checkpoint.keras', verbose=1, save_best_only=True)

decoder_inputs = Input(shape=(latent_dim,))
decoded = Dense(32)(decoder_inputs)
decoded = LeakyReLU(alpha=0.2)(decoded)
decoded = Dense(64)(decoded)
decoded = LeakyReLU(alpha=0.2)(decoded)
decoded = Dense(128)(decoded)
decoded = LeakyReLU(alpha=0.2)(decoded)
decoder_outputs = Dense(original_dim)(decoded)

vae = Model(inputs=decoder_inputs, outputs=decoder_outputs)
vae.summary()

vae.compile(optimizer='adam', loss='mse')
results = vae.fit(in_train, out_train,
                      shuffle=True,
                      batch_size = batch_size, 
                      epochs = epochs,
                      validation_data = (in_valid,out_valid),
                      callbacks = checkpointer)

vae.save("C:\\Users\\keena\\Documents\\University of Arizona\\Jobs\\TIMESTEP NOIRLAB\\wise-agn\\autoencoder\\model_decoder.keras")