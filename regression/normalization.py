import numpy as N
import pylab as p
import h5py
import sys
sys.path.append('')
from interpolation import LogInterpolate
import filters


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


def get_filters(filterlib='/home/robert/science/sedfit/filters.hdf5',\
                filternames=('wise-w1-3.4-r', 'wise-w2-4.6-r', 'wise-w3-12-r', 'wise-w4-22-r')):

    from scipy import integrate
    import sys
    sys.path.append('/home/robert/science/sedfit')
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
        ipt = LogInterpolate(wave,sed)  # filter interpolator
        
        # calculate Vega-normalized fuilter fluxes for the model, and write to HDF file
        # F_clumpy(W1)/F_vega(W1), F_clumpy(W2)/F_vega(W2), F_clumpy(W3)/F_vega(W3), F_clumpy(W4)/F_vega(W4)
        for ifilt,filt in enumerate(filters):
            
            # torus SED
            fclumpyt_ip = ipt(filt.lam)   # clumpy model flux on the filter's wavelength grid
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



