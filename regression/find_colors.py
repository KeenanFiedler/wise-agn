import numpy as N
import h5py as h5
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import filters
from keras.models import load_model
path = "C:\\Users\\keena\\Documents\\University of Arizona\\Jobs\\TIMESTEP NOIRLAB\\wise-agn\\"

def LogInterpolate(zz, xx, yy):
    logz = np.log10(zz)
    logx = np.log10(xx)
    logy = np.log10(yy)
    return np.power(10.0, np.interp(logz, logx, logy))

def flambda_bb(lam,T):
    """lam = wavelength in micron, T = BB temperature in K."""

    c = 2.99792458e+8    # speed of light [m/s]
    h = 6.62606896e-34   # Planck constant [J*s]
    k = 1.3806504e-23    # Boltzmann constant [J/K]

    lam_ = lam * 1.e-6

    aux1 = (2. * h * c) / (lam_**5.)
    aux2 = 1. / (N.exp( h * c / (lam_ * k * T) ) - 1.)
    B_lam = aux1 * aux2

    return B_lam


def flambda_vega(lam):
    """Vega flux F_lambda according to Eq. 2 in Wright et al. 2010"""

    flam_vega = 1.0158e-16 * (1. - 0.0083 * N.log(lam/8.891)**2.) * flambda_bb(lam,14454.)

    return flam_vega


def get_filters(filterlib=path+"regression\\filters.hdf5",
                filternames=('wise-w1-3.4-r', 'wise-w2-4.6-r', 'wise-w3-12-r', 'wise-w4-22-r')):

    from scipy import integrate
    import filters

    # filter lib
    flib = filters.FilterLib(filterlib)
    filters = flib.get(filternames,normalization='raw')

    # pre-calculate Vega normalizations for the 4 filters
    vega_normalizations = []
    for filt in filters:
        aux = integrate.simps(filt.phi * filt.lam * flambda_vega(filt.lam), filt.lam)
        vega_normalizations.append(aux)

    return filters, vega_normalizations


def get_filterfluxes(filters,vega_normalizations,wave,seds):

    """
    If seds is a 1-dim array, return a 1-d arrays of Nfilternames
    filterfluxes.

    If seds is 2-dimensional, assume that dim(0) is an SED index, and
    dim(1) contains the SED of each model, i.e. if seds.shape =
    (30,119), then seds[14,:] is the SED of model with index 14 (out
    of 30 SEDs contained in 'seds'.) Then, return a list of
    one-dimensional arrays:

    [FF1,FF2,...,FFN]

    with FF1 = array(filterflux1_sed1,filterflux1_sed2,...)

    etc.

    """

    from scipy import integrate

    nseds = seds.shape[0]
    filterfluxes = N.zeros((nseds,len(filters)))

    for ised in range(nseds):
        sed = seds[ised,:]
        log_wave = N.log10(wave)
        log_sed = N.log10(sed)
        lin_interp = sp.interpolate.interp1d(log_wave, log_sed)
        #ipt = LogInterpolate(wave,sed)  # filter interpolator
        
        # calculate Vega-normalized fuilter fluxes for the model, and write to HDF file
        # F_clumpy(W1)/F_vega(W1), F_clumpy(W2)/F_vega(W2), F_clumpy(W3)/F_vega(W3), F_clumpy(W4)/F_vega(W4)
        for ifilt,filt in enumerate(filters):
            
            # torus SED
            fclumpyt_ip = N.power(10.0, lin_interp(N.log10(filt.lam)))
            #fclumpyt_ip = ipt(filt.lam)   # clumpy model flux on the filter's wavelength grid
            filtfluxt = integrate.simps(filt.phi * fclumpyt_ip, filt.lam)
            filterfluxes[ised,ifilt] = filtfluxt / vega_normalizations[ifilt]

    return filterfluxes


def get_colors(filterfluxes):

    nseds = filterfluxes.shape[0]
    ncolors = filterfluxes.shape[1] - 1

    colors = N.zeros((nseds,ncolors))

    for j in range(ncolors):
        colors[:,j] = 2.5*N.log10(filterfluxes[:,j+1]/filterfluxes[:,j])

    return colors

def generate_seds(n_sed, min_array = [5.0,1.0,0.0,0.0,15.0,10.0], max_array = [100.0,15.0,1.0,3.0,70.0,300.0]):
    # Draw random values in between the minimum and maximum parameters, of size requested
    random_draws = N.random.uniform(min_array, max_array, size=(n_sed,6))

    # load machine learning model and find SEDs
    model = load_model(path + 'autoencoder\\3layer_64\\model_decoder_gpu_64.keras')
    fluxes = 10**model.predict(random_draws)

    return fluxes

def generate_colortrack(n_sed, wave, min_array = [5.0,1.0,0.0,0.0,15.0,10.0], max_array = [100.0,15.0,1.0,3.0,70.0,300.0]):
    # Draw random values in between the minimum and maximum parameters, of size requested
    random_draws = N.random.uniform(min_array, max_array, size=(n_sed,6))

    #Set up
    colortrack_array = N.zeros((n_sed,50,2))
    filters, vega_norm = get_filters()

    for j in range(n_sed):
        #Set up array of parameters with evenly distributed cos(i)
        cosi = N.linspace(0,1,50)
        i = N.degrees(N.arccos(cosi))
        model_array = N.zeros((50,6))
        model_array[:, 0] = random_draws[j][0]
        model_array[:, 1] = random_draws[j][1]
        model_array[:, 2] = i
        model_array[:, 3:] = random_draws[j][3:]

        # Load and draw from model
        model = load_model(path + 'autoencoder\\3layer_64\\model_decoder_gpu_64.keras')
        fluxes = 10**model.predict(model_array)

        # Save to plot SED as function of i
        N.save('colortrack_flux.npy', fluxes)

        # Get colors
        colors = get_colors(get_filterfluxes(filters, vega_norm, wave, fluxes))
        w21 = colors[:,0]
        w32 = colors[:,1]

        # Append colortrack to full array of color tracks
        c = w21.reshape(len(w21),1)
        c = N.append(c, w32.reshape(len(w32),1),1)
        colortrack_array[j] = c
    
    N.save('colortracks.npy', colortrack_array)


def main():
    infile = h5.File(path + "pca\\clumpy_models_201410_tvavg.hdf5", 'r')
    wave = infile['wave'][:]

    filters, vega_norm = get_filters()
    fluxes = generate_seds(10**6) #Generate seds using the machine learning model, 1 million seds in <15 seconds

    colors = get_colors(get_filterfluxes(filters, vega_norm, wave, fluxes))
    N.save('model_colors.npy', colors)

    generate_colortrack(100, wave)

main()
