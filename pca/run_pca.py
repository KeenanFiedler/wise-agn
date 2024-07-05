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
wave_norm = dset['wave'][:]
wave = np.log10(dset['wave'][:])


n_comp = [7,10,15,20]#range(7,21)

#output arrays
err = []
err_mean = []
compression = []
max_arr= []
mse = []
outliers = []

for amt in n_comp:

    #PCA Analysis
    pca = PCA(n_components=amt)
    mu = np.mean(dataset, axis=0)
    pca.fit(dataset)
    weights = pca.transform(dataset)

    #Rebuilding Spectra
    Xhat = np.dot(weights, pca.components_)
    Xhat += mu
    Xhat = 10**Xhat

    #Error Analysis
    err_set_all = np.abs((Xhat-flux))/flux
    mse_one = np.array(np.mean(np.square(err_set_all), axis=1))
    mse.append(mse_one)
    spectra = np.argmax(mse_one)
    outliers.append(Xhat[spectra])
    outliers.append(flux[spectra])
    max_arr.append(np.max(err_set_all))
    err.append(np.median(err_set_all))
    err_mean.append(np.mean(err_set_all))
    compression.append(1/((weights.size+pca.components_.size)/dataset.size))
    print(amt)

"""
j=0
arr = [7,10,15,20]
for i in [0, 6, 16, 26]:
    err_set_all = (outliers[i]-outliers[i+1])/outliers[i+1]
    fig = plt.figure()
    gs  = gridspec.GridSpec(2, 1, height_ratios=[3,1])
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1],sharex=ax0)
    plt.subplots_adjust(wspace=0, hspace=0)
    ax0.loglog(wave_norm, outliers[i], 'r-', label='Reconstructed')
    ax0.loglog(wave_norm, outliers[i+1], 'b--', label='Original')
    ax1.plot(wave_norm, err_set_all)
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
    ax0.title.set_text(str(arr[j]) + ' Component Reconstruction of Spectra')
    plt.xlabel('Log Wavelength (Microns)')
    ax0.set_ylabel('Log Flux')
    ax1.set_ylabel('Error')

    plt.savefig('Spectra_comparison_' + str(arr[j]) + '_outlier_test.png', dpi=300, bbox_inches='tight')
    j += 1
    plt.clf()

"""

#Plotting
_, bins, _ = plt.hist(mse[0], bins=50, range=[0, 0.004], density = True, alpha = 0.3, label = str(n_comp[0]) + ' Components')
for i in range(1,len(n_comp)):
    plt.hist(mse[i], bins = bins, density = True, alpha = 0.3, label = str(n_comp[i]) + ' Components')
plt.legend()
plt.xlim(-0.0004,0.004)
plt.title('Histogram of errors')
plt.xlabel('Mean Square Error')
plt.savefig('mse.png',dpi=300)
plt.clf()
"""
plt.plot(n_comp,max_arr)
plt.title('Error Curve')
plt.xlabel('Number of Components')
plt.ylabel('Max Error across all Spectra')
plt.xticks(range(6,22,2)) 
plt.savefig('max.png',dpi=300)
plt.clf()

plt.plot(n_comp, err)
plt.title('Error Curve')
plt.xlabel('Number of Components')
plt.ylabel('Median Error across all Spectra')
plt.xticks(range(6,22,2)) 
plt.savefig('error.png', dpi = 300)
plt.clf()

plt.plot(n_comp, err_mean)
plt.title('Error Curve')
plt.xlabel('Number of Components')
plt.ylabel('Mean Error across all Spectra')
plt.xticks(range(6,22,2)) 
plt.savefig('error_mean.png', dpi = 300)
plt.clf()

plt.plot(n_comp, compression)
plt.title('Compression Curve')
plt.xlabel('Number of Components')
plt.ylabel('Compression (1:val)')
plt.xticks(range(6,22,2)) 
plt.savefig('compression.png', dpi = 300)
plt.clf()
"""
#############################
#For Plotting Single Spectra#
#############################
"""
for amt in n_comp:
    pca = PCA(n_components=amt)
    mu = np.mean(dataset, axis=0)
    pca.fit(dataset)
    weights = pca.transform(dataset)
    Xhat = np.dot(weights, pca.components_)
    Xhat += mu
    Xhat = 10**Xhat

    err_set_all = (Xhat[1000000]-flux[1000000])/flux[1000000]
    fig = plt.figure()
    gs  = gridspec.GridSpec(2, 1, height_ratios=[3,1])
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1],sharex=ax0)
    plt.subplots_adjust(wspace=0, hspace=0)
    ax0.loglog(wave_norm, Xhat[1000000], 'r-', label='Reconstructed')
    ax0.loglog(wave_norm, flux[1000000], 'b--', label='Original')
    ax1.plot(wave_norm, err_set_all)
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
    ax0.title.set_text(str(amt) + ' Component Reconstruction of Spectra')
    plt.xlabel('Log Wavelength (Microns)')
    ax0.set_ylabel('Log Flux')
    ax1.set_ylabel('Error')

    plt.savefig('Spectra_comparison_' + str(amt) + '_1000000.png', dpi=300, bbox_inches='tight')
    plt.clf()



recon = Xhat[:,:111]
orig = dataset[:,:111]
print(np.median(np.abs(((recon-orig)/orig))*100))
print(np.median(np.abs(((Xhat-dataset)/dataset))*100))

#print('Shape of components array: ' + str(pca.components_.shape))
#print('Array of explained variance for each component: ' + str(pca.explained_variance_ratio_))
#print('Total explained variance by the components: ' + str(sum(pca.explained_variance_ratio_)))
"""