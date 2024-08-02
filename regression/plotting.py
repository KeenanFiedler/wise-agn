import numpy as np
import h5py as h5
import scipy as sp
import pandas as pd
import geopandas as gpd

import matplotlib.pyplot as plt
import matplotlib as mpl

import filters

from keras.models import load_model

import shapely
from shapely.geometry import Point, LineString
from shapely.geometry.polygon import Polygon

arr = np.load('ct_p_coeff.npz')

coeff = arr['coeff'][:]
params = arr['params'][:]
arr = None

Y = params[:,0]
N0 = params[:,1]
i = params[:,2]
q = params[:,3]
sig = params[:,4]
tv = params[:,5]

fig, ax = plt.subplots(3,2)

ax[0,0].hist(Y, bins = 20, density = True, label = 'Original', color='k', histtype='step')
ax[0,0].hist(Y, weights=coeff, bins = 20, density = True, label = 'Weighted', color = 'g', histtype='step')
ax[0,0].set_xlabel('Y')
ax[0,1].hist(N0, bins = 20, density = True, label = 'Original', color='k', histtype='step')
ax[0,1].hist(N0, weights=coeff, bins = 20, density = True, label = 'Weighted', color = 'g', histtype='step')
ax[0,1].set_xlabel('N0')
ax[1,0].hist(i, bins = 20, density = True, label = 'Original', color='k', histtype='step')
ax[1,0].hist(i, weights=coeff, bins = 20, density = True, label = 'Weighted', color = 'g', histtype='step')
ax[1,0].set_xlabel('i')
ax[1,1].hist(q, bins = 20, density = True, label = 'Original', color='k', histtype='step')
ax[1,1].hist(q, weights=coeff, bins = 20, density = True, label = 'Weighted', color = 'g', histtype='step')
ax[1,1].set_xlabel('q')
ax[2,0].hist(sig, bins = 20, density = True, label = 'Original', color='k', histtype='step')
ax[2,0].hist(sig, weights=coeff, bins = 20, density = True, label = 'Weighted', color = 'g', histtype='step')
ax[2,0].set_xlabel('sigma')
ax[2,1].hist(tv, bins = 20, density = True, label = 'Original', color='k', histtype='step')
ax[2,1].hist(tv, weights=coeff, bins = 20, density = True, label = 'Weighted', color = 'g', histtype='step')
ax[2,1].set_xlabel('tv')


handles, labels = ax[0,0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=2)
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.subplots_adjust(bottom=0.15)
fig.suptitle('Parameters vs weighted versions')
plt.savefig('test_hist.png', bbox_inches = 'tight',dpi=500)

