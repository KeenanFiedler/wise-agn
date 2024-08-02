import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py as h5
import shapely
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
path = "C:\\Users\\keena\\Documents\\University of Arizona\\Jobs\\TIMESTEP NOIRLAB\\wise-agn\\"


"""
t1_vert = [(2.5,1.0),(2.5,1.5),(3.5,1.5),(3.5,1.0),(2.5,1.0)]
t2_vert = [(4.0,2.0),(4.0,3.0),(5.0,3.0),(5.0,2.0),(4.0,2.0)]

colors = np.load('clumpy_colors.npy')
w21 = colors[:,0]
w23 = colors[:,1]
ind = np.where((w21 > 2.2099) & (w23 > 5.89336))[0]
w21 = w21[ind]
print('Number of AGN in CLUMPY outside of range of model generator: ' + str(len(w21)))
"""
def plot_colors(filename):
    colors = np.load(filename)
    w21 = colors[:,0]
    w23 = colors[:,1]
    plt.hist2d(w23,w21, bins = (50,50), norm=mpl.colors.LogNorm(), cmap=plt.cm.jet)
    #t1_poly = Polygon(t1_vert)
    #t2_poly = Polygon(t2_vert)
    #plt.plot(*t1_poly.exterior.xy, 'r:', label = 'Type 1 AGN')
    #plt.plot(*t2_poly.exterior.xy, 'k:', label = 'Type 2 AGN')
    plt.colorbar()
    plt.ylabel('W1-W2')
    plt.xlabel('W2-W3')
    plt.xlim(1.25,5.5)
    plt.ylim(0.5,3)
    plt.legend()
    plt.title('Histogram of Colors')
    plt.savefig(filename[:-4] + '_hist2d.png', dpi=300, bbox_inches = 'tight')
    plt.clf()

# 2-D Histogram
plot_colors('clumpy_colors.npy')
#plot_colors('model_colors.npy')

"""
# Color Tracks
colortracks = np.load('colortracks.npy')
plasma = mpl.colormaps['plasma'].resampled(len(colortracks))
k=0
for track in colortracks:
    plt.plot(track[:,1], track[:,0],color= plasma(k/len(colortracks)))
    k+=1
plt.ylabel('W2-W1')
plt.xlabel('W2-W3')
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
"""
