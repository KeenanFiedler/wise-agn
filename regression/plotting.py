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

arr = np.load('ct_p_coeff_big_0.1.npz')

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

ax[0,0].hist(Y, bins = 20, density = True, label = 'Original', color='b', histtype='step')
ax[0,0].hist(Y, weights=coeff, bins = 20, density = True, label = 'Weighted', color = 'k', histtype='step')
ax[0,0].set_xlabel('Y')
handles, labels = ax[0,0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=2)
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.subplots_adjust(bottom=0.15)
fig.suptitle('Parameters vs weighted versions')
plt.savefig('test_hist.png', bbox_inches = 'tight',dpi=500)

