import numpy as np
import h5py as h5
import scipy as sp
import pandas as pd
import geopandas as gpd
import pylab as p

import matplotlib.pyplot as plt
import matplotlib as mpl

from numpy import inf


import filters

from keras.models import load_model

import shapely
from shapely.geometry import Point, LineString
from shapely.geometry.polygon import Polygon

path = "C:\\Users\\keena\\Documents\\University of Arizona\\Jobs\\TIMESTEP NOIRLAB\\wise-agn\\"

def find_y_grid(width):
    w21_min, w21_max = 0.5,3
    w32_min, w32_max = 1.25,5.5

    range_ = [[w32_min,w32_max],[w21_min,w21_max]]

    cx = 0
    cy = 0
    grid_cells = []
    for y0 in np.arange(w21_min, w21_max+width, width):
        cy += 1
        for x0 in np.arange(w32_min, w32_max+width, width):
            x1 = x0+width
            y1 = y0+width
            new_cell = shapely.geometry.box(x0, y0, x1, y1)
            grid_cells.append(new_cell)
            if y0 == 0.5:
                cx += 1

    print(cx)
    print(cy)
    desi = pd.read_csv(path + 'regression\\agn_explore\\desi.csv')
    sdss = pd.read_csv(path + 'regression\\agn_explore\\sdss.csv')
    w1 = np.array(sdss.w1)
    w1 = np.append(w1, desi.w1)
    w2 = np.array(sdss.w2)
    w2 = np.append(w2, desi.w2)
    w3 = np.array(sdss.w3)
    w3 = np.append(w3, desi.w3)
    xarr = w2-w3
    yarr = w1-w2
    H,_,_ = np.histogram2d(xarr,yarr,bins=[cx,cy],range=range_)
    return H

"""
test = np.load('ENET_small.npz')
print(test.files)
a = test['alphas'][:]
l = test['l1s'][:]
grid = test['mse'][:,:,2]
print(np.unravel_index(grid.argmin(), grid.shape))
plt.pcolormesh(a, l, grid, cmap='jet')
plt.colorbar()
plt.scatter(a[8], l[87], color = 'r', label = 'Minimum MSE')
plt.xlabel(r'α Regularization')
plt.ylabel(r'L1 Ratio')
plt.title('Validation Mean Square Error for Parameter Range')
plt.savefig('val_curve.png', dpi=500, bbox_inches = 'tight')
plt.clf()

plt.plot(test['alphas'][:-2], test['dual_gaps'][:-2])
plt.plot(test['alphas'][:], np.mean(test['mse'][:,:,2],axis = 1))
plt.show()
plt.clf()
plt.plot(test['l1s'][:-2], test['dual_gaps'][:-2])
plt.plot(test['l1s'][:], np.mean(test['mse'][:,:,2],axis = 0))
plt.show()
plt.clf()
"""

H = find_y_grid(0.05)

arr = np.load('data\\ct_p_coeff_big.npz')

coeff = arr['coeff'][:]
print(np.count_nonzero(coeff))
#X = arr['hit_miss'][:]
#data_pred_2d = np.sum(X*coeff, axis=1)
#data_pred_2d = data_pred_2d.reshape((51,86))
#np.save('data_pred.npy', data_pred_2d)
data_pred_2d = np.load('data\\data_pred.npy')
#plotting

fig = p.figure(figsize=(8/1.5,12/1.5))
cmap = p.cm.jet
norm = mpl.colors.LogNorm()
clabel = r'counts / mag$^2$'
intp = 'none'
def make_panel(ax,data,extent=None,title='',ylabel='',xlabel='', morm = mpl.colors.LogNorm()):
    im = ax.imshow(data,origin='lower',extent=extent,cmap=cmap,interpolation=intp,norm=norm)
    cb = p.colorbar(im)
    cb.set_label(clabel)
    p.title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

extent = [1.75,5.5,0.5,3]
cellarea = 0.05*0.05

ax1 = fig.add_subplot(311)
make_panel(ax1,H.T/cellarea,extent=extent,title='Data',ylabel='W1-W2')

ax2 = fig.add_subplot(312)
make_panel(ax2,data_pred_2d/cellarea,extent=extent,title='Elastic Net',ylabel='W1-W2')

ax3 = fig.add_subplot(313)
norm = mpl.colors.Normalize()

im = ax3.imshow((((data_pred_2d-H.T)/H.T)*100),origin='lower',extent=extent,cmap=p.cm.bwr,interpolation=intp,norm=norm)
cb = p.colorbar(im)
im.set_clim(vmin=-200, vmax=200)
cb.set_label(r'percent error')
p.title('Percent Error')
ax3.set_xlabel('W2-W3')
ax3.set_ylabel('W1-W2')


error = (np.abs(data_pred_2d-H.T)/H.T).flatten()
error[error == -inf] = -1
error[error == inf] = 1
error[~np.isfinite(error)] = 0
COLOR = 'red'
mpl.rcParams['text.color'] = COLOR
plt.tight_layout()
import matplotlib.patheffects as pe
ax3.text(3.625,2.8, 'Mean Abs Error = ' + str(round(np.mean(100*error),4)),fontsize=14, weight = 'bold', horizontalalignment='center', verticalalignment='center',path_effects=[pe.withStroke(linewidth=2, foreground="black")])
p.savefig('data_big.png', dpi=500, bbox_inches='tight')
p.clf()

"""
arr = np.load('p_coeff.npz')

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
fig.suptitle('Uniform and Weighted Parameters')
plt.savefig('test_hist.png', bbox_inches = 'tight',dpi=500)
"""

