import numpy as np
import h5py as h5
import scipy as sp
import pandas as pd
import geopandas as gpd

import matplotlib.pyplot as plt
import matplotlib as mpl

infile = h5.File("C:\\Users\\keena\\Documents\\University of Arizona\\Jobs\\TIMESTEP NOIRLAB\\wise-agn\\pca\\clumpy_models_201410_tvavg.hdf5", 'r')
Y = infile['Y'][:]
N0 = infile['N0'][:]
i = infile['i'][:]
q = infile['q'][:]
sig = infile['sig'][:]
tv = infile['tv'][:]

ind = np.where((Y==10) & (N0 == 7) & (q==1) & (sig==15) & (tv==300))[0]
Y = Y[ind]
N0 = N0[ind]
flux_tor = infile['flux_tor'][ind]
wave = infile['wave'][:]
i = i[ind]
q = q[ind]
sig = sig[ind]
tv = tv[ind]


plasma = mpl.colormaps['plasma'].resampled(len(flux_tor))
for i in range(len(flux_tor)):
    plt.loglog(wave,flux_tor[i], color = plasma(i/len(flux_tor)))

plt.title('Example Spectra')
plt.xlabel('Log Wavelength (Microns)')
plt.ylabel('Log Flux')
plt.savefig('example.png', dpi=1000, bbox_inches='tight')