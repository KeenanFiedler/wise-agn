import numpy as np
import h5py as h5
import numpy as np 
import math
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
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

print(model.history.history)
unique_test = np.array([7,1.5,25,1.75,43,250])
unique_test = unique_test.reshape((1,6))
unique_test = model(unique_test)
unique_test = 10**unique_test
below_spectra = output_flux[47593]
above_spectra = output_flux[171174]

unique_test = unique_test[0,:]
plt.loglog(wave, unique_test, 'r-', label = 'New Between Spectra')
plt.loglog(wave, below_spectra, 'b--', label = 'Below Values Spectra')
plt.loglog(wave, above_spectra,'g--', label = 'Above Values Spectra')
plt.legend()
plt.xlabel('Log Wavelength')
plt.ylabel('Log Flux')
plt.title('New Unique Spectra vs Old')
plt.savefig('New_generation.png', dpi=300)
plt.clf()


err_set_all = (vae_test[100]-out_test[100])/out_test[100]
fig = plt.figure()
gs  = gridspec.GridSpec(2, 1, height_ratios=[3,1])
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1],sharex=ax0)
plt.subplots_adjust(wspace=0, hspace=0)
ax0.loglog(wave, vae_test[100], 'r-',label = 'Generated Spectra')
ax0.loglog(wave, out_test[100], 'b--', label = 'True Spectra')
ax1.plot(wave, err_set_all)
ax1.set_xscale('log')

xlim0 = ax0.get_xlim()
ylim0 = ax0.get_ylim()
xlim1 = ax1.get_xlim()
ylim1 = ax1.get_ylim()

ax1.fill_between(range(10000), -0.05, 0.05, alpha=0.25, color = 'k', linewidth=0.0)

val = math.ceil(max(ylim1)*10)/10

ax0.set_xlim(xlim0)
ax0.set_ylim(ylim0)
ax1.set_xlim(xlim1)
ax1.set_ylim(val*-1,val)

ax0.legend()
ax0.title.set_text('Decoder Reconstruction of Spectra')
plt.xlabel('Log Wavelength (Microns)')
ax0.set_ylabel('Log Flux')
ax1.set_ylabel('Error')

plt.savefig('Spectra_comparison_decoder.png', dpi=300, bbox_inches='tight')
plt.clf()