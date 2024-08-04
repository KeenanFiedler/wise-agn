import numpy as np
import h5py as h5
import math
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib import gridspec

infile = h5.File("C:\\Users\\keena\\Documents\\University of Arizona\\Jobs\\TIMESTEP NOIRLAB\\wise-agn\\pca\\clumpy_models_201410_tvavg.hdf5", 'r')
dset = infile

#Setting up datasets
flux = np.array(dset['flux_tor'][:])
dataset = np.log10(np.array(dset['flux_tor'][:]))
wave = np.array(dset['wave'][:])

pca = PCA(n_components=10)
mu = np.mean(dataset, axis=0)
pca.fit(dataset)
weights = pca.transform(dataset)

#Rebuilding Spectra
Xhat = np.dot(weights, pca.components_)
Xhat += mu
Xhat = 10**Xhat

with h5.File('pca.h5', 'w') as hf:
    hf.create_dataset("weights",  data=weights)
    hf.create_dataset("components",  data=pca.components_)
    hf.create_dataset("mu",  data=mu)
    hf.create_dataset("wave",  data=wave)