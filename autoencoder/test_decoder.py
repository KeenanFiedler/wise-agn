import numpy as np
import h5py as h5
import numpy as np 
import math
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import scipy.stats
from sklearn.decomposition import PCA

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
path ="C:\\Users\\keena\\Documents\\University of Arizona\\Jobs\\TIMESTEP NOIRLAB\\wise-agn\\"

infile = h5.File(path + "pca\\clumpy_models_201410_tvavg.hdf5", 'r')
output_flux = infile['flux_tor'][:]
wave = infile['wave'][:]
Y = infile['Y'][:]
N0 = infile['N0'][:]
i = infile['i'][:]
q = infile['q'][:]
sig = infile['sig'][:]
tv = infile['tv'][:]
"""
infile = h5.File(path + "pca\\models_subset.hdf5", 'r')
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

model = tf.keras.models.load_model(path + 'autoencoder\\conv_32\\model_decoder_conv_32.keras')


hist = pd.read_csv(path + 'autoencoder\\conv_32\\history_conv_32.csv', delimiter=',')
print(hist)
epochs = range(1,101)[1:]
loss = hist['loss'][1:]
val_loss = hist['val_loss'][1:]
plt.plot(epochs,loss, label = 'Loss')
plt.plot(epochs, val_loss, label = 'Validation Loss')
plt.legend()
plt.savefig(path + 'autoencoder\\conv_32\\loss_conv_32.png', dpi = 300)
plt.clf()

accuracy = hist['accuracy'][1:]
val_accuracy = hist['val_accuracy'][1:]
plt.plot(epochs, accuracy, label = 'Accuracy')
plt.plot(epochs, val_accuracy, label = 'Validation Accuracy')
plt.legend()
plt.savefig(path + 'autoencoder\\conv_32\\accuracy_conv_32.png', dpi = 300)
plt.clf()

##### CREATE UNIQUE SPECTRA INBETWEEN

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
plt.savefig(path + 'autoencoder\\conv_32\\New_generation_conv_32.png', dpi=300)
plt.clf()


in_flux = model.predict(input_params)
in_flux = 10**in_flux

err_set_all = (in_flux-output_flux)/output_flux
mse = np.array(np.mean(np.square(err_set_all), axis=1))

##### PLOT MEAN SQUARE ERROR VERSUS PCA MSE

flux = np.array(infile['flux_tor'][:])
dataset = np.log10(np.array(infile['flux_tor'][:]))


pca = PCA(n_components=7)
mu = np.mean(dataset, axis=0)
pca.fit(dataset)
weights = pca.transform(dataset)

#Rebuilding Spectra
Xhat = np.dot(weights, pca.components_)
Xhat += mu
Xhat = 10**Xhat

#Error Analysis
err_set_PCA = np.abs((Xhat-flux))/flux
mse_PCA = np.array(np.mean(np.square(err_set_PCA), axis=1))

_, bins, _ = plt.hist(mse, bins = 30, range=[0, 0.004], density = True, alpha = 0.3, label = 'Decoder')
plt.hist(mse_PCA, bins = bins, density = True, alpha = 0.3, label = 'PCA - 7 Components')
plt.legend()
plt.xlabel('Mean Square Error')
plt.savefig(path + 'autoencoder\\conv_32\\mse_7_conv_32.png', dpi=300)
plt.clf()

##### CREATE PLOT OF SPECTRA WITH MOST ERROR
spectra = np.argmax(mse)
in_flux = in_flux[spectra]
output_flux = output_flux[spectra]
err_set_all = err_set_all[spectra]
fig = plt.figure()
gs  = gridspec.GridSpec(2, 1, height_ratios=[3,1])
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1],sharex=ax0)
plt.subplots_adjust(wspace=0, hspace=0)
ax0.loglog(wave,in_flux, 'r-',label = 'Generated Spectra')
ax0.loglog(wave, output_flux, 'b--', label = 'True Spectra')
ax1.plot(wave, err_set_all)
ax1.set_xscale('log')

xlim0 = ax0.get_xlim()
ylim0 = ax0.get_ylim()
xlim1 = ax1.get_xlim()
ylim1 = ax1.get_ylim()

ax1.fill_between(range(10000), -0.05, 0.05, alpha=0.25, color = 'k', linewidth=0.0)

if abs(ylim1[0]) > abs(ylim1[1]):
    val = math.floor(ylim1[0]*10)/10
else:
    val = math.ceil(ylim1[1]*10)/10


ax0.set_xlim(xlim0)
ax0.set_ylim(ylim0)
ax1.set_xlim(xlim1)
if val < 0:
    ax1.set_ylim(val,val*-1)
else:
    ax1.set_ylim(val*-1,val)

ax0.legend()
ax0.title.set_text('Decoder Reconstruction of Spectra')
plt.xlabel('Log Wavelength (Microns)')
ax0.set_ylabel('Log Flux')
ax1.set_ylabel('Error')

plt.savefig(path + 'autoencoder\\conv_32\\Spectra_comparison_max_err_conv_32.png', dpi=300, bbox_inches='tight')
plt.clf()


##### PLOT SPECIFIC TEST SPECTRA

err_set_all = (vae_test[10]-out_test[10])/out_test[10]
fig = plt.figure()
gs  = gridspec.GridSpec(2, 1, height_ratios=[3,1])
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1],sharex=ax0)
plt.subplots_adjust(wspace=0, hspace=0)
ax0.loglog(wave, vae_test[10], 'r-',label = 'Generated Spectra')
ax0.loglog(wave, out_test[10], 'b--', label = 'True Spectra')
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

plt.savefig(path + 'autoencoder\\conv_32\\Spectra_comparison_decoder_conv_32.png', dpi=300, bbox_inches='tight')
plt.clf()
