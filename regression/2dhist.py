import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py as h5
path = "C:\\Users\\keena\\Documents\\University of Arizona\\Jobs\\TIMESTEP NOIRLAB\\wise-agn\\"


def plot_colors(filename):
    colors = np.load(filename)
    w21 = colors[:,0]
    w32 = colors[:,1]
    plt.hist2d(w21,w32, bins = (50,50))
    plt.colorbar()
    plt.xlabel('W2-W1')
    plt.ylabel('W3-W2')
    plt.title('Colors of Histogram')
    plt.savefig(filename[:-4] + '_hist2d.png', dpi=300)
    plt.clf()

# 2-D Histogram
plot_colors('clumpy_colors.npy')
plot_colors('model_colors.npy')


# Color Tracks
colortracks = np.load('colortracks.npy')
plasma = mpl.colormaps['plasma'].resampled(len(colortracks))
k=0
for track in colortracks:
    plt.plot(track[:,0], track[:,1],color= plasma(k/len(colortracks)))
    k+=1
plt.xlabel('W2-W1')
plt.ylabel('W3-W2')
plt.title('100 Colortracks')
plt.savefig('colortracks.png',dpi=300)
plt.clf()


infile = h5.File(path + "pca\\clumpy_models_201410_tvavg.hdf5", 'r')
wave = infile['wave'][:]
fluxes = np.load('colortrack_flux.npy')
cosi = np.linspace(0,1,50)
inc = np.round(np.degrees(np.arccos(cosi)), decimals=2)


for i in [0,4,9,14,19,24,29,34,39,44,49]:
    plt.loglog(wave, fluxes[i], alpha=0.75, label = 'Inclination: ' + str(inc[i]))
plt.legend(loc = 3, prop={'size': 6})
plt.xlabel('Log Wavelength')
plt.ylabel('Log Flux')
plt.title('Evolution of SED as function of i')
plt.savefig('cosi_evolution_sed.png', dpi=300)
plt.clf()
