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


path = "C:\\Users\\keena\\Documents\\University of Arizona\\Jobs\\TIMESTEP NOIRLAB\\wise-agn\\"
wave = [0.01,0.015,0.022,0.033,0.049,0.073,0.1,0.123,0.151,0.185,0.227,0.279,0.343,0.421,0.517,
        0.55,0.635,0.78,0.958,1.18,1.45,1.77,2.18,2.68,3.29,4.04,4.96,6,6.25,6.5,6.75,7,7.25,7.5,
        7.75,8,8.25,8.5,8.6,8.7,8.8,8.9,9,9.1,9.2,9.3,9.4,9.5,9.6,9.7,9.8,9.9,10,10.1,10.2,10.3,10.4,
        10.5,10.6,10.7,10.8,10.9,11,11.1,11.2,11.3,11.4,11.5,11.8,12,12.3,12.5,12.8,13,13.3,13.5,13.8,
        14,14.3,14.5,14.8,15,15.3,15.5,15.8,16,16.3,16.5,16.8,17,17.3,17.5,17.8,18,18.3,18.5,18.8,19,
        19.3,19.5,19.8,20,20.9,25.6,31.5,38.7,47.5,58.3,71.6,87.9,108,133,163,200,204,304,452,672,1000]

""" HELPER FUNCTIONS """
def LogInterpolate(zz, xx, yy):
    logz = np.log10(zz)
    logx = np.log10(xx)
    logy = np.log10(yy)
    return np.power(10.0, np.interp(logz, logx, logy))

def points_in_polygon(x, y, vertices):
    points = gpd.GeoDataFrame(geometry = gpd.points_from_xy(x, y))
    polygon = Polygon(vertices)

    return shapely.contains(polygon, points)


""" MAGNITUDE FUNCTIONS """
def draw_normal(value, amount):
    if value == 'z':
        mu = 0.61274080
        sig = 0.26551053
    elif value == 'w1':
        mu = 15.1269908
        sig = 0.87389070
    
    return np.random.normal(mu, sig, amount)

def get_mags(n, filters, vega_norm, wave):
    w1 = draw_normal('w1', n)
    fluxes = generate_seds(n)
    colors = get_colors(get_filterfluxes(filters, vega_norm, wave, fluxes))
    w2 = w1 - colors[:,0]
    w3 = w2 - colors[:,1]
    w4 = w3 - colors[:,2]


""" FILTER FUNCTIONS """
def flambda_bb(lam,T):
    """lam = wavelength in micron, T = BB temperature in K."""

    c = 2.99792458e+8    # speed of light [m/s]
    h = 6.62606896e-34   # Planck constant [J*s]
    k = 1.3806504e-23    # Boltzmann constant [J/K]

    lam_ = lam * 1.e-6

    aux1 = (2. * h * c) / (lam_**5.)
    aux2 = 1. / (np.exp( h * c / (lam_ * k * T) ) - 1.)
    B_lam = aux1 * aux2

    return B_lam

def flambda_vega(lam):
    """Vega flux F_lambda according to Eq. 2 in Wright et al. 2010"""

    flam_vega = 1.0158e-16 * (1. - 0.0083 * np.log(lam/8.891)**2.) * flambda_bb(lam,14454.)

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
    filterfluxes = np.zeros((nseds,len(filters)))

    for ised in range(nseds):
        sed = seds[ised,:]
        log_wave = np.log10(wave)
        log_sed = np.log10(sed)
        lin_interp = sp.interpolate.interp1d(log_wave, log_sed)
        #ipt = LogInterpolate(wave,sed)  # filter interpolator
        
        # calculate Vega-normalized fuilter fluxes for the model, and write to HDF file
        # F_clumpy(W1)/F_vega(W1), F_clumpy(W2)/F_vega(W2), F_clumpy(W3)/F_vega(W3), F_clumpy(W4)/F_vega(W4)
        for ifilt,filt in enumerate(filters):
            
            # torus SED
            fclumpyt_ip = np.power(10.0, lin_interp(np.log10(filt.lam)))
            #fclumpyt_ip = ipt(filt.lam)   # clumpy model flux on the filter's wavelength grid
            filtfluxt = integrate.simps(filt.phi * fclumpyt_ip, filt.lam)
            filterfluxes[ised,ifilt] = filtfluxt / vega_normalizations[ifilt]

    return filterfluxes

def get_colors(filterfluxes):

    nseds = filterfluxes.shape[0]
    ncolors = filterfluxes.shape[1] - 1

    colors = np.zeros((nseds,ncolors))

    for j in range(ncolors):
        colors[:,j] = 2.5*np.log10(filterfluxes[:,j+1]/filterfluxes[:,j])

    return colors

""" GENERATING FUNCTIONS """
def generate_seds(n_sed, min_array = [5.0,1.0,0.0,0.0,15.0,10.0], max_array = [100.0,15.0,1.0,3.0,70.0,300.0]):
    # Draw random values in between the minimum and maximum parameters, of size requested
    random_draws = np.random.uniform(min_array, max_array, size=(n_sed,6))
    
    # load machine learning model and find SEDs
    model = load_model(path + 'autoencoder\\3layer_64\\model_decoder_gpu_64.keras')
    fluxes = 10**model.predict(random_draws)

    return fluxes

def generate_colortrack(n_sed, n_cos, wave, min_array = [5.0,1.0,0.0,0.0,15.0,10.0], max_array = [100.0,15.0,1.0,3.0,70.0,300.0]):
    # Draw random values in between the minimum and maximum parameters, of size requested
    random_draws = np.random.uniform(min_array, max_array, size=(n_sed,6))
    filters, vega_norm = get_filters()

    # Set up large array for quicker model drawing with n_sed*n_cos models
    cosi = np.linspace(0,1,n_cos) # linear spaced in cos(i)
    i = np.degrees(np.arccos(cosi))
    long_sed_params = np.repeat(random_draws, repeats=n_cos, axis=0)
    print(long_sed_params.shape)
    long_sed_params[:, 2] = np.tile(i, long_sed_params.shape[0] // i.shape[0])

    #Predict
    model = load_model(path + 'autoencoder\\3layer_64\\model_decoder_gpu_64.keras')
    fluxes = 10**model.predict(long_sed_params)
    print(fluxes.shape)
    #Split back into colortracks
    indices = np.arange(n_cos, fluxes.shape[0], n_cos)
    fluxes = np.array(np.array_split(fluxes, indices, 0))
    #fluxes = np.array(np.array_split(fluxes, n_cos))
    colortrack_array = np.zeros((n_sed,n_cos,2))

    for j in range(n_sed):
        flux_for_color = fluxes[j,:,:]
        # Save to plot SED as function of i
        #np.save('colortrack_flux.npy', flux_for_color)

        # Get colors
        colors = get_colors(get_filterfluxes(filters, vega_norm, wave, flux_for_color))
        w21 = colors[:,0]
        w32 = colors[:,1]

        # Append colortrack to full array of color tracks
        c = w21.reshape(len(w21),1)
        c = np.append(c, w32.reshape(len(w32),1),1)
        colortrack_array[j] = c
    
    return colortrack_array, random_draws


def get_models_in_polygon(t1_vert, t2_vert, n_sed, n_cos, width):
    """
    0) Initialize filters and output array
    Loop over (random, dynamically generated) models:
    1) Generate a random parameter vector (tv,N0,q,sig,Y) in generate_colortrack()
    2) Generate n_cos (n_cos=100 or so) random viewings to that model (uniform in cos(i)). in generate_colortrack()
    3) For all M random viewings generate an SED in generate_colortrack()
    4) For all M SEDs, calculate colors x=W2-W3, y=W1-W2 in generate_colortrack()
    5) Find all model+viewing combos which have x,y colors in the blue region (i.e. "IR-classified type-1 region").
       and simultaneously have viewings with colors in type-2 (red) region.
    6) Go to 1), and repeat N times (maybe a 1e5 or 1e6 times)
    """
    #Setup
    filters, vega_normalizations = get_filters()
    colortracks, params = generate_colortrack(n_sed,n_cos,wave)
    #Output Arrays
    agn_tracks = np.zeros((1,n_cos,2))
    agn_params = np.zeros((1,6))
    j=0
    k=0
    #Find all viewings which have colors in type1-region ('blue box'), and simultaneously in type2-region ('red box')
    indexes = []
    for i in range(colortracks.shape[0]):
        track = colortracks[i]
        param = params[i]
        x = track[:,1]
        y = track[:,0]

        # Get indexes of hit or miss matrix for this colortrack
        indexes.append(find_hit_or_miss(width, gpd.GeoDataFrame({'geometry':LineString(gpd.points_from_xy(x,y))}, index = [0], geometry = 'geometry')))
        
        #Check if the points of the color track are in the type1/2 boxes
        in_polygon = pd.DataFrame()
        in_polygon['t1'] = points_in_polygon(track[:,1], track[:,0], t1_vert)
        in_polygon['t2'] = points_in_polygon(track[:,1], track[:,0], t2_vert)

        #Plotting colortrack

        #If the track is in type 1 and type 2 boxes, add it to the list
        if (True in in_polygon['t1'].values) and (True in in_polygon['t2'].values):
            agn_tracks = np.append(agn_tracks, track[None, :, :], axis = 0)
            agn_params = np.append(agn_params, param[None,:], axis = 0)
            """
            if j == 0:
                plt.plot(track[:,1], track[:,0], 'g', label = 'AGN-like', lw=0.5)
                j+=1
            else:
                plt.plot(track[:,1], track[:,0], 'g', lw=0.5)
        else:
            if k == 0:
                plt.plot(track[:,1], track[:,0], 'r--', label = 'Not AGN-like', lw=0.5)
                k+=1
            else:
                plt.plot(track[:,1], track[:,0], 'r--', lw=0.5)
    """
    #Drawing polygons for plot
    """
    t1_poly = Polygon(t1_vert)
    t2_poly = Polygon(t2_vert)
    plt.plot(*t1_poly.exterior.xy, 'b:', label = 'Type 1 AGN')
    plt.plot(*t2_poly.exterior.xy, 'k:', label = 'Type 2 AGN')
    plt.legend()
    plt.xlabel('W2-W3')
    plt.ylabel('W1-W2')
    plt.title('Models whose evolution over viewing angle lies in Type 1 and Type 2')
    plt.savefig('polygons.png', dpi=300)
    plt.clf()
    """
    plot_grid(indexes, colortracks, width)
    #Cut out zero first element (Try to fix needing this step later)
    agn_tracks = agn_tracks[1:]
    agn_params = agn_params[1:]

    return agn_tracks, agn_params

def find_hit_or_miss(width,track):
    w21_min, w21_max = 0, 5
    w32_min, w32_max = 1, 10

    grid_cells = []
    for x0 in np.arange(w32_min, w32_max+width, width):
        for y0 in np.arange(w21_min, w21_max+width, width):
            x1 = x0-width
            y1 = y0+width
            new_cell = shapely.geometry.box(x0, y0, x1, y1)
            grid_cells.append(new_cell)
    
    grid_cells = gpd.GeoDataFrame(geometry=grid_cells)
    g = grid_cells.copy()
    joined = gpd.sjoin(track, g)['index_right'].unique().tolist()
    return joined

def plot_grid(indexes, colortracks, width):
    w21_min, w21_max = 0, 5
    w32_min, w32_max = 1, 10

    grid_cells = []
    for x0 in np.arange(w32_min, w32_max+width, width):
        for y0 in np.arange(w21_min, w21_max+width, width):
            x1 = x0-width
            y1 = y0+width
            new_cell = shapely.geometry.box(x0, y0, x1, y1)
            grid_cells.append(new_cell)
    
    grid_cells = gpd.GeoDataFrame(geometry=grid_cells)

    plasma = mpl.colormaps['plasma'].resampled(len(indexes))
    for i in range(len(indexes)):
        ind = indexes[i]
        track = colortracks[i]
        g = grid_cells.copy().loc[ind]
        for index, row in g.iterrows():
            square = row['geometry']
            plt.plot(*square.exterior.xy, color = plasma(i/len(indexes)), lw=0.1)
            plt.plot(track[:,1], track[:,0], 'g', lw=0.1)
    plt.xlabel('W2-W3')
    plt.ylabel('W1-W2')
    plt.title('Models and their intersection with the hit or miss grid')
    plt.savefig('boxes.png', dpi = 500)
    plt.clf()

def main():
    t1_vert = [(2.5,1.0),(2.5,1.5),(3.5,1.5),(3.5,1.0),(2.5,1.0)]
    t2_vert = [(4.0,2.0),(4.0,3.0),(5.0,3.0),(5.0,2.0),(4.0,2.0)]

    # Type1 vert, Type2 vert, n_sed, n_cosine, bin_width
    get_models_in_polygon(t1_vert,t2_vert,100,100,0.05)

    #mags = get_mags(10, filters, vega_norm, wave)
    #np.save('magnitudes.npy', mags)

    #fluxes = generate_seds(10) #Generate seds using the machine learning model, 1 million seds in <15 seconds

    #colors = get_colors(get_filterfluxes(filters, vega_norm, wave, fluxes))
    #np.save('model_colors.npy', colors)

    #generate_colortrack(100, wave)

main()
