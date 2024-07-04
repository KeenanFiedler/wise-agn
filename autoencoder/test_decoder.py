import numpy as np
import h5py as h5
import numpy as np 
import matplotlib.pyplot as plt
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


infile = h5.File("C:\\Users\\keena\\Documents\\University of Arizona\\Jobs\\TIMESTEP NOIRLAB\\wise-agn\\pca\\clumpy_models_201410_tvavg.hdf5", 'r')
output_flux = infile['flux_tor'][:]
wave = infile['wave'][:]
Y = infile['Y'][:]
N0 = infile['N0'][:]
i = infile['i'][:]
q = infile['q'][:]
sig = infile['sig'][:]
tv = infile['tv'][:]
"""
infile = h5.File("C:\\Users\\keena\\Documents\\University of Arizona\\Jobs\\TIMESTEP NOIRLAB\\wise-agn\\pca\\models_subset.hdf5", 'r')
output_flux = infile['dset']['flux_tor'][:]
wave = infile['dset']['wave'][0]
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

model = tf.keras.models.load_model('C:\\Users\\keena\\Documents\\University of Arizona\\Jobs\\TIMESTEP NOIRLAB\\wise-agn\\autoencoder\\model_checkpoint.keras')


vae_test = model.predict(in_test)
vae_test = 10**vae_test
plt.loglog(wave, vae_test[0], 'r-',label = 'Generated Spectra')
plt.loglog(wave, out_test[0], 'b--', label = 'True Spectra')
plt.legend()
plt.savefig('test_spectra.png', dpi=300)
plt.clf()