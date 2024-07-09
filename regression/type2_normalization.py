import numpy as N
import pylab as p
import h5py
import sys
sys.path.append('/home/robert/science/sedfit')
from interpolation import LogInterpolate
import filters


def histOutline(dataIn, *args, **kwargs):
    
    import numpy as N

    (histIn, binsIn) = N.histogram(dataIn, *args, **kwargs)

    stepSize = binsIn[1] - binsIn[0]

    bins = N.zeros(len(binsIn)*2 + 2, dtype=N.float)
    data = N.zeros(len(binsIn)*2 + 2, dtype=N.float)
    for bb in range(len(binsIn)):
        bins[2*bb + 1] = binsIn[bb]
        bins[2*bb + 2] = binsIn[bb] + stepSize
        if bb < len(histIn):
            data[2*bb + 1] = histIn[bb]
            data[2*bb + 2] = histIn[bb]

    bins[0] = bins[1]
    bins[-1] = bins[-2]
    data[0] = 0
    data[-1] = 0

    return (bins, data)


def fnu_bb(nu,T):
    """lam = wavelength in micron, T = BB temperature in K."""

    c = 2.99792458e+8    # speed of light [m/s]
    h = 6.62606896e-34   # Planck constant [J*s]
    k = 1.3806504e-23    # Boltzmann constant [J/K]

#    lam_ = lam * 1.e-6
#    nu = c/lam_

    aux1 = (2. * h * nu**3) / (c**2.)
    aux2 = 1. / (N.exp( h * nu / (k * T) ) - 1.)
    B_nu = aux1 * aux2

    return B_nu


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


def newfunc(dbfile='/fast/robert/data/clumpy/models/20130415_tvavg_resampled.hdf5',t1box=(1.,1.5,2.5,3.5),t2box=(2.,3.,4.,5.)):

    """
    0) Load CLUMPY DB, and instantiate an interpolation object.
    Loop over (random, dynamically generated) models:
    1) Generate a random parameter vector (tv,N0,q,sig,Y) (with PyMC maybe?)
#    2) Calculate it's f2.
    2) Generate M (M=100 or 1000 or so) random viewings to that model (uniform in cos(i)).
    3) For all M viewings calculate Pesc.
    4) For all M random viewings, interpolate an SED. Yields M SEDs.
    5) For all M SEDs, calculate colors x=W2-W3, y=W1-W2.
    6) Find all model+viewing combos which have x,y colors in the blue region (i.e. "IR-classified type-1 region").
       and simultaneously have viewings with colors in type-2 (red) region. Calculate mean filterfluxes for those t1 and t2 viewings.

#       Take avg. cos(i) and avg. Pesc for those models.
    7) Calculate the ration R1, R2, R3 = Ri(t1) / Ri(t2) for all these fluxes
    8) Go to 1), and repeat N times (maybe a 1e5 or 1e6 times)

    """

    import sys
    sys.path.append('/home/robert/science/sedfit')
    import ndinterpolation
    import pymc
                
    n = 1000   # number of random models
    m = 100    # number of random viewings per model

    # 0)
    print "Instantiating SED interpolator object..."
    ip = ndinterpolation.LocIntp(dbfile,type='torus',order=1,mode='log')

    # 1)
    # npar-dimensional uniform random number generator; get new number by calling u.rand()
    # no viewing included in this object, will be separate issue.
    low = [par[0] for par in ip.theta[1:-1]]
    up  = [par[-1] for par in ip.theta[1:-1]]
    ug = pymc.Uniform('u',low,up)
    print "ug.parents: ", ug.parents
#    sys.exit()

    filters, vega_normalizations = get_filters()

    Dm = []

    for jn in range(n):
        r_ = ug.rand()
        print "r_ = ", r_

        # 2)
        # CAUTION: the viewings generated here are uniform in
        # cos(i). The 'ip' interpolator object however works only
        # correctly if it is fed with 1.-v values. Executive summary:
        # feed ip with 1.-cos(i), but in our book-keeping keep the
        # result associated with the cos(i) value.
        v = N.random.uniform(0.,1.,size=m)  # viewing; cos(i) random number generator; uniform, 1-dimensional

        # 3)
        # calculate Pesc for all viewings
        # tbd

        # 4)
        # interpolate SEDs for all viewings
        seds = N.zeros((m,ip.wave.size))
        for jv in range(m):
#            seds[jv,:] = ip(N.concatenate(([v[jv]],r_)),ip.wave)
            seds[jv,:] = ip(N.concatenate(([1.0-v[jv]],r_)),ip.wave)  # See the CAUTION under 2)!!!

        # 5)
        # For all SEDs for this model, calculate filterfluxes in W1--W3, and colors W1-W2 & W2-W3
        filterfluxes = get_filterfluxes(filters, vega_normalizations,ip.wave,seds)
        colors = get_colors(filterfluxes)  # CAUTION: colors.shape = (n,2), and colors[:,0] = W1-W2, colors[:,1] = W2-W3

        # 6)
        # For this model, find all viewings which have colors in type1-region ('blue box'), and simultaneously in type2-region ('red box')
        t1_indices_true, t1_indices_false = points_in_polygon(colors,t1box)
        if t1_indices_true.size == 0:
            continue  # skip this model if no viewings with colors in t1-box found

        t2_indices_true, t2_indices_false = points_in_polygon(colors,t2box)
        if t2_indices_true.size == 0:
            continue  # skip this model if no viewings with colors in t2-box found

        # we only get here if both t1_indices_true and t2_indices_true aren't empty
        filterfluxes_t1 = filterfluxes[t1_indices_true,:]
        filterfluxes_t2 = filterfluxes[t2_indices_true,:]

        filterfluxes_t1_mean = filterfluxes_t1.mean(axis=0)  # mean model flux for all viewings with compatible t1 colors
        filterfluxes_t2_mean = filterfluxes_t2.mean(axis=0)  # mean model flux for all viewings with compatible t2 colors

        # 7)
        # Calculate Dm values for current model
        Dm.append( 2.5*N.log10(filterfluxes_t1_mean/filterfluxes_t2_mean) )  # 1-D array of 3 numbers (this is for the current model, all viewings involved)


#    print "Generating %d random models (no viewing angles yet)..." % n
#    r = N.array([u.rand() for j in range(n)])

    return ip, v, seds, filterfluxes, colors, N.array(Dm)
    
    
def points_in_polygon(x,y,polygon,polygontype='rectangle'):

    """
    CAUTION: For now, polygon is strictly a rectangle, with [xmin,xmax,ymin,ymax]\

    Take an array of N points, with 2 colors provided for every point,
    and return the indices for which colors lie inside of the provided
    polygon.

    Example:

    colors = [[1.1,2.0],   # [[x1,y1,],
              [0.5,1.0],      [x2,y2,],
              [0.1,1.7],      [x3,y3,],
              [1.0,1.5]]      [x4,y4,]]

    polygon = [0.6,2.0,1.3,2.5]  # [xmin,xmax,ymin,ymax]


    Diagram: x & y axes, polygon-box, and data points (dots). Two dots
    are inside the box, two are outside.


         y |  
           |    2.5 |---------------|
           |        |     .         |
           |  .     |               |
           |        |    .          |
           |    1.3 |---------------|
           |       0.6             2.0
           |      .
           |  
           --------------------------------------
                                         x
    
    This function returns the indices of the inside-the-box points,
    i.e. (0,3), and of the outside-the-box points, i.e. (1,2).

    """

    if polygontype == 'circle':  # (w12,w23,radius), i.e. a circle
        xc, yc, dr = polygon
        r = N.sqrt( (x-xc)**2. + (y-yc)**2. )
        co = (r <= dr)
    elif polygontype == 'rectangle':  # (w12.min,w12.max,w23.min,w23.max), i.e. rectangle
        xl,xr,yl,yr = polygon
        co = (x >= xl) & (x < xr) & (y >= yl) & (y < yr)
    elif polygontype == 'ellipse':
        cx,cy,rx,ry = polygon   # x of points, y of points, x-center of ellipse, y-center of ellipse, ellipse x-semiaxis, ellipse y-semiaxis
        dr = ((x-cx)/rx)**2. + ((y-cy)/ry)**2.
        co = (dr <= 1.)
    else:
        raise Exception, "Unkown polygontype"
            
    indices_true = N.argwhere(co)  #[:,0]
#    indices_false = N.argwhere(~co)

    return indices_true #, indices_false

#def points_in_polygon(colors,polygon):
#
#    """
#    CAUTION: For now, polygon is strictly a rectangle, with [xmin,xmax,ymin,ymax]\
#
#    Take an array of N points, with 2 colors provided for every point,
#    and return the indices for which colors lie inside of the provided
#    polygon.
#
#    Example:
#
#    colors = [[1.1,2.0],   # [[x1,y1,],
#              [0.5,1.0],      [x2,y2,],
#              [0.1,1.7],      [x3,y3,],
#              [1.0,1.5]]      [x4,y4,]]
#
#    polygon = [0.6,2.0,1.3,2.5]  # [xmin,xmax,ymin,ymax]
#
#
#    Diagram: x & y axes, polygon-box, and data points (dots). Two dots
#    are inside the box, two are outside.
#
#
#         y |  
#           |    2.5 |---------------|
#           |        |     .         |
#           |  .     |               |
#           |        |    .          |
#           |    1.3 |---------------|
#           |       0.6             2.0
#           |      .
#           |  
#           --------------------------------------
#                                         x
#    
#    This function returns the indices of the inside-the-box points,
#    i.e. (0,3), and of the outside-the-box points, i.e. (1,2).
#
#    """
#
#    c12 = colors[:,0]
#    c23 = colors[:,1]
#
#    if len(polygon) == 3:  # (w12,w23,radius), i.e. a circle
#        r = N.sqrt( (c12-polygon[0])**2. + (c23-polygon[1])**2. )
#        dr = 2.*polygon[2]
#        co = (r <= dr)
#    elif len(polygon) == 4:  # (w12.min,w12.max,w23.min,w23.max), i.e. rectangle
#        co = (c12 > polygon[0]) & (c12 < polygon[1]) & (c23 > polygon[2]) & (c23 < polygon[3])
#
#    indices_true = N.argwhere(co)
#    indices_false = N.argwhere(~co)
#
#    return indices_true, indices_false


def get_Dm(dbfile,t1box,t2box):

    """
    Default: dbfile = '/fast/robert/data/clumpy/models/20130326_tvavg_resampled.hdf5'

    """

    import h5py
    import numpy as N

    hc = h5py.File(dbfile,'r')
    hco = h5py.File('/fast/robert/data/clumpy/models/20120609_tvavg.hdf5','r')
    f_2 = hco['f2'][:]
    hco.close()

    # CLUMPY model quantitites
    gc = hc['clumpy_parameters']
    nmodels = gc['q'].shape[0]
    pesc = gc['pesc'][:]

    # filterfluxes (normalized to Vega)
    gf = hc['filterfluxes_veganormalized_torus']
    f1 = gf['wise-w1-3.4-r'][:]
    f2 = gf['wise-w2-4.6-r'][:]
    f3 = gf['wise-w3-12-r'][:]

    q  = gc['q'][:]
    N0 = gc['N0'][:]
    tavg = gc['tavg'][:]
    Y = gc['Y'][:]
    sig = gc['sig'][:]
    i = gc['i'][:]

#    hc.close()
#    return f1, f2, f3
    

    w12 = 2.5*N.log10(f2/f1)
    w23 = 2.5*N.log10(f3/f2)

    Dm1, Dm2, Dm3 = [], [], []

    ql, N0l, tavgl, Yl, sigl, i1l, i2l, pesc1l, pesc2l, f21l, f22l = [], [], [], [], [], [], [], [], [], [], []

    startidx = N.arange(nmodels)[::10]
    for jidx, idx in enumerate(startidx):
        if idx % 1000 == 0: print "%d of %d (%.2f percent)" % (jidx,startidx.size,100.*jidx/float(startidx.size))

        idxes = N.arange(nmodels)[idx:idx+10]
        pesc_ = pesc[idxes]
        if pesc_.size != 10:
            raise Exception, "pesc_.size = ", pesc_.size
        w12_ = w12[idxes]
        w23_ = w23[idxes]
        f1_ = f1[idxes]
        f2_ = f2[idxes]
        f3_ = f3[idxes]

        q_ = q[idxes]
#        if q_1.sum()/float(q_1.size) != q_1[0]:
#            raise Exception, "Not all q values are the same"
#        else:
#            ql.append(q_1[0])

        N0_ = N0[idxes]
#        if N0_1.sum()/float(N0_1.size) != N0_1[0]:
#            raise Exception, "Not all N0 values are the same"
#        else:
#            N0l.append(N0_1[0])

        tavg_ = tavg[idxes]
#        if tavg_1.sum()/float(tavg_1.size) != tavg_1[0]:
#            raise Exception, "Not all tavg values are the same"
#        else:
#            tavgl.append(tavg_1[0])

        Y_ = Y[idxes]
#        if Y_1.sum()/float(Y_1.size) != Y_1[0]:
#            raise Exception, "Not all Y values are the same"
#        else:
#            Yl.append(Y_1[0])

        sig_ = sig[idxes]
#        if sig_1.sum()/float(sig_1.size) != sig_1[0]:
#            raise Exception, "Not all sig values are the same"
#        else:
#            sigl.append(sig_1[0])

        i_ = i[idxes]
#        if jcosi_.sum()/float(jcosi_.size) != jcosi_[0]:
#            raise Exception, "Not all jcosi values are the same"
#        else:
#            jcosil.append(jcosi_[0])

        pesc_ = pesc[idxes]
        f_2_ = f_2[idxes]

#        cot1 = (pesc_ > 0.5)
#        cot2 = ~cot1

#        cot1 = (w12_ > 1.0) & (w12_ < 1.5) & (w23_ > 2.5) & (w23_ < 3.5)
#        cot2 = (w12_ > 2.0) & (w12_ < 3.0) & (w23_ > 4.0) & (w23_ < 5.0)
        cot1 = (w12_ > t1box[0]) & (w12_ < t1box[1]) & (w23_ > t1box[2]) & (w23_ < t1box[3])
        cot2 = (w12_ > t2box[0]) & (w12_ < t2box[1]) & (w23_ > t2box[2]) & (w23_ < t2box[3])


        f1_t1_avg = f1_[cot1].mean()
        if f1_[cot1].size == 0:
#            print "f1_[cot1].size = ", f1_[cot1].size
#            print "jidx, idx", jidx, idx
            continue
        f1_t2_avg = f1_[cot2].mean()
        if f1_[cot2].size == 0:
            print "f1_[cot2].size = ", f1_[cot2].size
            print "jidx, idx", jidx, idx
            continue

        f2_t1_avg = f2_[cot1].mean()
        if f2_[cot1].size == 0:
#            print "f2_[cot1].size = ", f2_[cot1].size
#            print "jidx, idx", jidx, idx
            continue
        f2_t2_avg = f2_[cot2].mean()
        if f2_[cot2].size == 0:
            print "f2_[cot2].size = ", f2_[cot2].size
            print "jidx, idx", jidx, idx
            continue

        f3_t1_avg = f3_[cot1].mean()
        if f3_[cot1].size == 0:
#            print "f3_[cot1].size = ", f3_[cot1].size
#            print "jidx, idx", jidx, idx
            continue
        f3_t2_avg = f3_[cot2].mean()
        if f3_[cot2].size == 0:
            print "f3_[cot2].size = ", f3_[cot2].size
            print "jidx, idx", jidx, idx
            continue

#        jcosil.append(list(jcosi_))

        ql.append(q_[cot1].mean())
        N0l.append(N0_[cot1].mean())
        tavgl.append(tavg_[cot1].mean())
        Yl.append(Y_[cot1].mean())
        sigl.append(sig_[cot1].mean())
        i1l.append(i_[cot1].mean())
        i2l.append(i_[cot2].mean())
        pesc1l.append(pesc_[cot1].mean())
        pesc2l.append(pesc_[cot2].mean())
        f21l.append(f_2_[cot1].mean())
        f22l.append(f_2_[cot2].mean())

        R1 = f1_t1_avg / f1_t2_avg
        R2 = f2_t1_avg / f2_t2_avg
        R3 = f3_t1_avg / f3_t2_avg
    
        Dm1_ = 2.5*N.log10(R1)
        Dm2_ = 2.5*N.log10(R2)
        Dm3_ = 2.5*N.log10(R3)

        Dm1.append(Dm1_)
        Dm2.append(Dm2_)
        Dm3.append(Dm3_)


    Dm1 = N.array(Dm1)
    Dm2 = N.array(Dm2)
    Dm3 = N.array(Dm3)

    hc.close()

    return Dm1, Dm2, Dm3,  ql, N0l, tavgl, Yl, sigl, i1l, i2l, pesc1l, pesc2l, f21l, f22l


def get_Dm_iDB(dbfile,t1box,t2box):

    """
    Default: dbfile = '/fast/robert/data/clumpy/models/20130326_tvavg_resampled.hdf5'

    """

    import h5py
    import numpy as N

    hc = h5py.File(dbfile,'r')

    # CLUMPY model quantitites
    gc = hc['clumpy_parameters']
    nmodels = gc['q'].shape[0]
#    pesc = gc['pesc'][:]
    vega_normalizations = hc['vega_normalizations'][:]

    # filterfluxes (normalized to Vega)
    gf = hc['filterfluxes_torus']
    f1 = gf['wise-w1-3.4-r'][:] / vega_normalizations[0]
    f2 = gf['wise-w2-4.6-r'][:] / vega_normalizations[1]
    f3 = gf['wise-w3-12-r'][:] / vega_normalizations[2]

#    hc.close()
#    return f1, f2, f3
    

    w12 = 2.5*N.log10(f2/f1)
    w23 = 2.5*N.log10(f3/f2)

    Dm1, Dm2, Dm3 = [], [], []

    startidx = N.arange(nmodels)[::10]
    for jidx, idx in enumerate(startidx):
        if idx % 1000 == 0: print "%d of %d (%.2f percent)" % (jidx,startidx.size,100.*jidx/float(startidx.size))

        idxes = N.arange(nmodels)[idx:idx+10]
#        pesc_ = pesc[idxes]
#        if pesc_.size != 10:
#            raise Exception, "pesc_.size = ", pesc_.size
        w12_ = w12[idxes]
        w23_ = w23[idxes]
        f1_ = f1[idxes]
        f2_ = f2[idxes]
        f3_ = f3[idxes]

#        cot1 = (pesc_ > 0.5)
#        cot2 = ~cot1

#        cot1 = (w12_ > 1.0) & (w12_ < 1.5) & (w23_ > 2.5) & (w23_ < 3.5)
#        cot2 = (w12_ > 2.0) & (w12_ < 3.0) & (w23_ > 4.0) & (w23_ < 5.0)
        cot1 = (w12_ > t1box[0]) & (w12_ < t1box[1]) & (w23_ > t1box[2]) & (w23_ < t1box[3])
        cot2 = (w12_ > t2box[0]) & (w12_ < t2box[1]) & (w23_ > t2box[2]) & (w23_ < t2box[3])


        f1_t1_avg = f1_[cot1].mean()
        if f1_[cot1].size == 0:
#            print "f1_[cot1].size = ", f1_[cot1].size
#            print "jidx, idx", jidx, idx
            continue
        f1_t2_avg = f1_[cot2].mean()
        if f1_[cot2].size == 0:
            print "f1_[cot2].size = ", f1_[cot2].size
            print "jidx, idx", jidx, idx
            continue

        f2_t1_avg = f2_[cot1].mean()
        if f2_[cot1].size == 0:
#            print "f2_[cot1].size = ", f2_[cot1].size
#            print "jidx, idx", jidx, idx
            continue
        f2_t2_avg = f2_[cot2].mean()
        if f2_[cot2].size == 0:
            print "f2_[cot2].size = ", f2_[cot2].size
            print "jidx, idx", jidx, idx
            continue

        f3_t1_avg = f3_[cot1].mean()
        if f3_[cot1].size == 0:
#            print "f3_[cot1].size = ", f3_[cot1].size
#            print "jidx, idx", jidx, idx
            continue
        f3_t2_avg = f3_[cot2].mean()
        if f3_[cot2].size == 0:
            print "f3_[cot2].size = ", f3_[cot2].size
            print "jidx, idx", jidx, idx
            continue

        R1 = f1_t1_avg / f1_t2_avg
        R2 = f2_t1_avg / f2_t2_avg
        R3 = f3_t1_avg / f3_t2_avg
    
        Dm1_ = 2.5*N.log10(R1)
        Dm2_ = 2.5*N.log10(R2)
        Dm3_ = 2.5*N.log10(R3)

        Dm1.append(Dm1_)
        Dm2.append(Dm2_)
        Dm3.append(Dm3_)


    Dm1 = N.array(Dm1)
    Dm2 = N.array(Dm2)
    Dm3 = N.array(Dm3)

    hc.close()

    return Dm1, Dm2, Dm3


def main(clumpy='/home/robert/science/wise_colors/type2_normalization/Dmnew2.npz',t1box=(1.,1.5,2.5,3.5),t2box=(2.,3.,4.,5.)):
#def main(clumpy='/home/robert/science/wise_colors/type2_normalization/Dmnew2.npz',t1box=(1.,1.5,2.5,3.5),t2box=(1.5,2.,4.5,5.5)):

    """ 
    
    clumpy : 'calculate' or filepath
    t1box : (t1_xmin, t1_xmax, t1_ymin, t1_ymax)
    t2box : (t2_xmin, t2_xmax, t2_ymin, t2_ymax)

    """


    ###########################################
    ### nType1(W1) & nType2(W1) (both data) ###
    # res1, res2 are tuples of 3 mags each (W1,W2,W3) for the sources from type1 and type2 regions, resp.
    ###########################################
#    res1 = load_raw_data('/home/robert/science/wise_colors/type2_normalization/t1region_zi__wise_allsky.wise_allsky_4band_p3as_psd26432.tbl')
##    res2 = load_raw_data('/home/robert/science/wise_colors/type2_normalization/t2region_zi__wise_allsky.wise_allsky_4band_p3as_psd3886.tbl')
#    res2 = load_raw_data('/home/robert/science/wise_colors/type2_normalization/t2region__rnmodified_wise_allsky.wise_allsky_4band_p3as_psd24043.tbl')

    data = load_raw_data('/home/robert/science/wise_colors/type2_normalization/t1t2regions__0.8_w12_3.0__2.0_w23_6.0__wise_allsky.wise_allsky_4band_p3as_psd27436.tbl')


    ###########################################
    # myres1, myres2 are after cleaning the samples; i.e. each ist a tuple of 3 arrays (W1,W2,W3), with Wx a 1-D array
    ###########################################
#    myres1, myres2 = get_samples(res1,res2)
    myres1, myres2 = get_samples(data,t1box,t2box)

#    fig1 = p.figure()
#    plot_nW1(myres1,myres2)
#    p.savefig('nTypeW1.pdf')


    ###########################################
    ### Dm1, Dm2, Dm2 FROM CLUMPY MODELS ###
    ###########################################
#    F0c, F90c = get_clumpy_fluxratios_classic(from_npzfile='/home/robert/science/wise_colors/type2_normalization/F0_F90.npz')
#    Dm1 = 2.5 * N.log10(F0c[:,0]/F90c[:,0])
#    Dm2 = 2.5 * N.log10(F0c[:,1]/F90c[:,1])
#    Dm3 = 2.5 * N.log10(F0c[:,2]/F90c[:,2])

#    fi = N.load('/home/robert/science/wise_colors/type2_normalization/Dm.npz')

    if clumpy == 'calculate':
        Dm1, Dm2, Dm3,   ql, N0l, tavgl, Yl, sigl, i1l, i2l, pesc1l, pesc2l, f21l, f22l = get_Dm('/fast/robert/data/clumpy/models/20130326_tvavg_resampled.hdf5',t1box,t2box) 
#        Dm1, Dm2, Dm3 = get_Dm_iDB('/home/robert/science/wise_colors/clumpy_wise_filterfluxes_20120910_w2+bb0.39.hdf5',t1box,t2box)
        fi = open('/home/robert/science/wise_colors/type2_normalization/Dm_2ndbox.npz','w')
#        fi = open('/home/robert/science/wise_colors/type2_normalization/Dmnew2_rnmodified.npz','w')
#        N.savez(fi,Dm1=Dm1,Dm2=Dm2,Dm3=Dm3)
        N.savez(fi,Dm1=Dm1,Dm2=Dm2,Dm3=Dm3,   q=ql, N0=N0l, tavg=tavgl, Y=Yl, sig=sigl, i1=i1l, i2=i2l, pesc1=pesc1l, pesc2=pesc2l, f21=f21l, f22=f22l)
        fi.close()
    elif clumpy != 'calculate':
        # '/home/robert/science/wise_colors/type2_normalization/Dmnew2.npz'
        fi = N.load(clumpy)  # color-color selected types
        Dm1 = fi['Dm1']
        Dm2 = fi['Dm2']
        Dm3 = fi['Dm3']

        fii = N.load('/home/robert/science/wise_colors/type2_normalization/Dm_iDB.npz')
        Dm1i = fii['Dm1']
        Dm2i = fii['Dm2']
        Dm3i = fii['Dm3']


#    print "Dm1.size, Dm1i.size = ", Dm1.size, Dm1i.size

#    print "Dm1.min(), Dm2.min(), Dm3.min() = ", Dm1.min(), Dm2.min(), Dm3.min()
#    print "Dm1.max(), Dm2.max(), Dm3.max() = ", Dm1.max(), Dm2.max(), Dm3.max()

    wm1 = weighted_mean(Dm1)
    wm2 = weighted_mean(Dm2)
    wm3 = weighted_mean(Dm3)


    ###############################################
    # PLOT Dm1, 2, 3 distribution of CLUMPY models
    ###############################################
    fig2 = p.figure(figsize=(6.,4.))
    mx = max(Dm1.max(),Dm2.max(),Dm3.max())
    normed = True

    c1i, b1i, dum = p.hist(Dm1i,bins=50,range=(0.,mx),histtype='step',color='b',lw=2,alpha=0.3,normed=normed,log=0)
    c2i, b2i, dum = p.hist(Dm2i,bins=50,range=(0.,mx),histtype='step',color='g',lw=2,alpha=0.3,normed=normed,log=0)
    c3i, b3i, dum = p.hist(Dm3i,bins=50,range=(0.,mx),histtype='step',color='r',lw=2,alpha=0.3,normed=normed,log=0)

    c1, b1, dum = p.hist(Dm1,bins=50,range=(0.,mx),histtype='step',color='b',lw=2,normed=normed,log=0,label='Dm1')
    c2, b2, dum = p.hist(Dm2,bins=50,range=(0.,mx),histtype='step',color='g',lw=2,normed=normed,log=0,label='Dm2')
    c3, b3, dum = p.hist(Dm3,bins=50,range=(0.,mx),histtype='step',color='r',lw=2,normed=normed,log=0,label='Dm3')

    print c3
    print b3

    p.axvline(Dm1.mean(),color='b',ls='-',lw=2)
    p.axvline(Dm2.mean(),color='g',ls='-',lw=2)
    p.axvline(Dm3.mean(),color='r',ls='-',lw=2)

#    p.axvline(wm1,color='b',ls='--',lw=2)
#    p.axvline(wm2,color='g',ls='--',lw=2)
#    p.axvline(wm3,color='r',ls='--',lw=2)

    p.axvline(N.median(Dm1),color='b',ls='-.',lw=2)
    p.axvline(N.median(Dm2),color='g',ls='-.',lw=2)
    p.axvline(N.median(Dm3),color='r',ls='-.',lw=2)


    p.xlabel('Dm1,Dm2,Dm3')
    p.ylabel('histogram')
    p.legend(loc='upper right',frameon=False)
#    p.title(r'Dm = 2.5$\cdot$log F(0deg)/F(90 deg)')
    p.title('%d CLUMPY models w/ t1 & t2 colors matching the prescription' % Dm1.size)
    p.grid()
    p.savefig('Dm123_cosi_alldata.pdf')
#    p.savefig('Dm123_zi_iDB.pdf')
#    p.savefig('Dm123_rnmodified.pdf')

    ###################
    # PREDICTED nType2
    ###################
    t1w1, t1w2, t1w3 = myres1
    print "t1w1.size (orig) = ", t1w1.size

    print "Dm1.mean(), Dm2.mean(), Dm3.mean() = ", Dm1.mean(), Dm2.mean(), Dm3.mean()
#    print "wm1, wm2, wm3 = ", wm1, wm2, wm3

#    t2w1 = t1w1 + Dm1.mean()
#    t2w2 = t1w2 + Dm2.mean()
#    t2w3 = t1w3 + Dm3.mean()

    t2w1 = t1w1 + wm1
    t2w2 = t1w2 + wm2
    t2w3 = t1w3 + wm3

#    t2w1 = t1w1 + N.median(Dm1)
#    t2w2 = t1w2 + N.median(Dm2)
#    t2w3 = t1w3 + N.median(Dm3)

    print "t2w1.size (added Dm_) = ", t2w1.size

    co = (t2w1 < 16.5) & (t2w2 < 15.5) & (t2w3 < 11.2)
    t2w1 = t2w1[co]  
    print "t2w1.size (after mag limits) = ", t2w1.size
    t2w2 = t2w2[co]
    t2w3 = t2w3[co]

    c12 = t2w1 - t2w2
    c23 = t2w2 - t2w3
    print "c12.min(), c12.max(), c12: ", c12.min(), c12.max(), c12
    print "c23.min(), c23.max(), c23: ", c23.min(), c23.max(), c23

#    cot2 = (c12 > 2.0) & (c12 < 3.0) & (c23 > 4.0) & (c23 < 5.0)
    cot2 = (c12 >= t2box[0]) & (c12 <= t2box[1]) & (c23 >= t2box[2]) & (c23 <= t2box[3])

    t2w1 = t2w1[cot2]  # we probably only need this one, i.e. nType2(W1b)_predicted
    t2w2 = t2w2[cot2]  # we probably only need this one, i.e. nType2(W1b)_predicted
    t2w3 = t2w3[cot2]  # we probably only need this one, i.e. nType2(W1b)_predicted
    print "t2w1.size (must be in t2 color space) = ", t2w1.size


#    fig0 = p.figure()
#    p.hist(c12,label='W1-W2')
#    p.hist(c23,label='W2-W3')
#    p.legend(loc='upper left',frameon=False)
#    p.savefig('colors_hists.pdf')


    # PLOT histograms of nType1(W1)_data, nType2(W1)_data, nType2(W1)_prediction
    fig1 = p.figure()
#    plot_nW1(myres1,myres2,t2w1)
    plot_nW1(myres1,myres2,(t2w1,t2w2,t2w3))
#    p.savefig('nTypeW1_zi_cosi_mean.pdf')
    p.savefig('nTypeW1_cosi_wavg_alldata_iautest1.pdf')
##    p.savefig('nTypeW1_cosi_wavg_lin.pdf')
#    p.savefig('nTypeW1_zi_cosi_median.pdf')
#    p.savefig('nTypeW1_zi_iDB.pdf')
#    p.savefig('nTypeW1_rnmodified.pdf')

    return myres1, myres2, Dm1, Dm2, Dm3,  c1, b1,   ql, N0l, tavgl, Yl, sigl, i1l, i2l, pesc1l, pesc2l, f21l, f22l


def converter(a):

    try:
        res = float(a)
    except:
        res = -12345.

    return res


def load_raw_data(fi,usecols=(12,16,20,7),skiprows=148):  # W1, W2, W3,  b

    converters = dict(zip(usecols,[converter]*len(usecols)))

    print "Loading raw columns..."
    res = N.loadtxt(fi,unpack=1,usecols=usecols,skiprows=skiprows)

    print "Defining initial co..."
    co = N.ones(res[0].size,dtype=N.bool)
    print "co.size: ", co.size

    print "Looping over columns, accumulating clean selections..."
    for i,u in enumerate(usecols):
        co = co & (res[i] != -12345.)
    
    print "Applying clean selection..."
    for i in range(len(res)):
        res[i] = res[i][co]
        print "res[i].size: ", res[i].size

    return res


#def get_samples(rest1,rest2,blim=10.,w1_brightlim=8.1,w1t1_faintlim=14.7,w1t2_faintlim=16.5):
#Rdef get_samples(data,t1box,t2box,blim=10.,w1_brightlim=8.1,w1t1_faintlim=14.7,w1t2_faintlim=16.5,polygontype='rectangle'):
def get_samples(data,t1box,t2box,blim=10.,w1_brightlim=8.1,w1t1_faintlim=14.82,w1t2_faintlim=16.83,polygontype='rectangle'):

    w1, w2, w3, b = data
    c12 = w1 - w2
    c23 = w2 - w3


#R    cot1 = points_in_polygon(c23,c12,t1box,polygontype)
#R    cot2 = points_in_polygon(c23,c12,t2box,polygontype)

    cot1 = points_in_polygon(c12,c23,t1box,polygontype)
    cot2 = points_in_polygon(c12,c23,t2box,polygontype)

#    if len(t1box) == 3: # circle  (w12,w23,radius)
#        r = N.sqrt( (c12-t1box[0])**2. + (c23-t1box[1])**2. )
#        dr = t1box[2]
#        cot1 = (r <= dr)
#    elif len(t1box) == 4:  # (w12.min,w12.max,w23.min,w23.max), i.e. rectangle
#        if boxtype == 'rectangle':
#            cot1 = (c12 >= t1box[0]) & (c12 <= t1box[1])  & (c23 >= t1box[2]) & (c23 <= t1box[3])
#        elif boxtype == 'ellipse':
#            cot1 = points_in_ellipse(c23,c12,t1box[0],t1box[1],t1box[2],t1box[3])  # x,y,xc,yc,rx,ry
#            print "GET_SAMPLES: in 'ellipse' branch, cot1[cot1==True].size = ", cot1[cot1==True].size
#
#    cot2 = (c12 >= t2box[0]) & (c12 <= t2box[1])  & (c23 >= t2box[2]) & (c23 <= t2box[3])

    w1t1, w2t1, w3t1, bt1 = w1[cot1], w2[cot1], w3[cot1], b[cot1]
    w1t2, w2t2, w3t2, bt2 = w1[cot2], w2[cot2], w3[cot2], b[cot2]

#    w1t1, w2t1, w3t1, bt1 = rest1
#    w1t2, w2t2, w3t2, bt2 = rest2
    
    print
    print "GET_SAMPLES: w1t1.size (orig) : ", w1t1.size
    print "GET_SAMPLES: w1t2.size (orig): ", w1t2.size
    print

    # GALACTIC LATITUDE LIMITS
    co = (N.abs(bt1) > blim)
    w1t1 = w1t1[co]
    w2t1 = w2t1[co]
    w3t1 = w3t1[co]

    co = (N.abs(bt2) > blim)
    w1t2 = w1t2[co]
    w2t2 = w2t2[co]
    w3t2 = w3t2[co]

    print
#R    print "GET_SAMPLES: w1t1.size (blim>10deg) : ", w1t1.size
#R    print "GET_SAMPLES: w1t2.size (blim>10deg): ", w1t2.size
    print "GET_SAMPLES: w1t1.size (blim>%d deg): " % blim, w1t1.size
    print "GET_SAMPLES: w1t2.size (blim>%d deg): " % blim, w1t2.size
    print

    # SATURATED/BRIGHT SOURCE LIMIT (IN W1 BAND)
    co = (w1t1 > w1_brightlim)
    w1t1 = w1t1[co]
    w2t1 = w2t1[co]
    w3t1 = w3t1[co]

    co = (w1t2 > w1_brightlim)
    w1t2 = w1t2[co]
    w2t2 = w2t2[co]
    w3t2 = w3t2[co]

    print
#R    print "GET_SAMPLES: w1t1.size (w1_brightlim<8.1mag) : ", w1t1.size
#R    print "GET_SAMPLES: w1t2.size (w1_brightlim<8.1mag): ", w1t2.size
    print "GET_SAMPLES: w1t1.size (w1_brightlim<%.2f mag): " % w1_brightlim, w1t1.size
    print "GET_SAMPLES: w1t2.size (w1_brightlim<%.2f mag): " % w1_brightlim, w1t2.size
    print

    # LIMITING W1 MAGNITUDE
    co = (w1t1 < w1t1_faintlim)
    w1t1 = w1t1[co]
    w2t1 = w2t1[co]
    w3t1 = w3t1[co]

    co = (w1t2 < w1t2_faintlim)
    w1t2 = w1t2[co]
    w2t2 = w2t2[co]
    w3t2 = w3t2[co]

    print
#R    print "GET_SAMPLES: w1t1.size (w1t1_faintlim=14.7) : ", w1t1.size
#R    print "GET_SAMPLES: w1t2.size (w1t2_faintlim=16.5): ", w1t2.size
    print "GET_SAMPLES: w1t1.size (w1t1_faintlim=%.2f mag): " % w1t1_faintlim, w1t1.size
    print "GET_SAMPLES: w1t2.size (w1t2_faintlim=%.2f mag): " % w1t2_faintlim, w1t2.size
    print

    myrest1 = (w1t1,w2t1,w3t1)
    myrest2 = (w1t2,w2t2,w3t2)

    return myrest1, myrest2


def weighted_mean(Dm):

    c,b = N.histogram(Dm,bins=50,range=(0.,5.0))
    locs = N.digitize(Dm,b)
    weights = [b[l] for l in locs]
    weighted_mean = N.average(Dm,weights=weights)

    return weighted_mean
    

def plot_nW1(res1,res2,nType2_pred):

    import pylab as p
    import numpy as N

    fontsize = 8.
#    fig_size = (4,6)
    p.rcParams['axes.labelsize'] = fontsize
    p.rcParams['text.fontsize'] =  fontsize
    p.rcParams['legend.fontsize'] = fontsize
    p.rcParams['xtick.labelsize'] = fontsize
    p.rcParams['ytick.labelsize'] = fontsize
#    p.rcParams['figure.figsize'] = fig_size
#    p.rcParams['font.family'] = 'serif'
    p.rcParams['font.family'] = 'sans-serif'

    w1t1, w2t1, w3t1 = res1
    w1t2, w2t2, w3t2 = res2

    w1_nType2_pred, w2_nType2_pred, w3_nType2_pred = nType2_pred

    w1_min_ = N.floor(min(w1t1.min(),w1t2.min(),w1_nType2_pred.min()))
    w1_max_ = N.ceil(max(w1t1.max(),w1t2.max(),w1_nType2_pred.max()))

    w2_min_ = N.floor(min(w2t1.min(),w2t2.min(),w2_nType2_pred.min()))
    w2_max_ = N.ceil(max(w2t1.max(),w2t2.max(),w2_nType2_pred.max()))

    w3_min_ = N.floor(min(w3t1.min(),w3t2.min(),w3_nType2_pred.min()))
    w3_max_ = N.ceil(max(w3t1.max(),w3t2.max(),w3_nType2_pred.max()))

    dmag = 0.2
    logscale = True

    fig = p.figure(figsize=(4,7.))
    
    ax1 = fig.add_subplot(311)
    p.hist(w1t1,bins=N.arange(w1_min_,w1_max_+dmag,dmag),range=(w1_min_,w1_max_),histtype='step',color='b',lw=2,log=logscale,label='nType1(W1), n = %d' % w1t1.size)
    p.hist(w1t2,bins=N.arange(w1_min_,w1_max_+dmag,dmag),range=(w1_min_,w1_max_),histtype='step',color='r',lw=2,log=logscale,label='nType2(W1), n = %d' % w1t2.size)
    p.hist(w1_nType2_pred,bins=N.arange(w1_min_,w1_max_+dmag,dmag),range=(w1_min_,w1_max_),histtype='step',color='y',lw=2,log=logscale,label='nType2(W1)_pred, n = %d' % w1_nType2_pred.size)
    p.xlabel('W1 (mag)')
    p.ylabel('n(W1) per %.2f mag bin' % dmag)
    p.legend(loc='upper left',frameon=False)

    ax2 = fig.add_subplot(312)
    p.hist(w2t1,bins=N.arange(w2_min_,w2_max_+dmag,dmag),range=(w2_min_,w2_max_),histtype='step',color='b',lw=2,log=logscale,label='nType1(W2)')
    p.hist(w2t2,bins=N.arange(w2_min_,w2_max_+dmag,dmag),range=(w2_min_,w2_max_),histtype='step',color='r',lw=2,log=logscale,label='nType2(W2)')
    p.hist(w2_nType2_pred,bins=N.arange(w2_min_,w2_max_+dmag,dmag),range=(w2_min_,w2_max_),histtype='step',color='y',lw=2,log=logscale,label='nType2(W2)_pred')
    p.xlabel('W2 (mag)')
    p.ylabel('n(W2) per %.2f mag bin' % dmag)
    p.legend(loc='upper left',frameon=False)

    ax3 = fig.add_subplot(313)
    p.hist(w3t1,bins=N.arange(w3_min_,w3_max_+dmag,dmag),range=(w3_min_,w3_max_),histtype='step',color='b',lw=2,log=logscale,label='nType1(W3)')
    p.hist(w3t2,bins=N.arange(w3_min_,w3_max_+dmag,dmag),range=(w3_min_,w3_max_),histtype='step',color='r',lw=2,log=logscale,label='nType2(W3)')
    p.hist(w3_nType2_pred,bins=N.arange(w3_min_,w3_max_+dmag,dmag),range=(w3_min_,w3_max_),histtype='step',color='y',lw=2,log=logscale,label='nType2(W3)_pred')
    p.xlabel('W3 (mag)')
    p.ylabel('n(W3) per %.2f mag bin' % dmag)
    p.legend(loc='upper left',frameon=False)

    p.subplots_adjust(bottom=0.05,top=0.985,left=0.12,right=0.975,hspace=0.22)

def get_clumpy_fluxratios(clumpydb='/fast/robert/data/clumpy/models/20120609_tvavg.hdf5',filterwaves=(3.4,4.6,12.)):

    """This is the ndinterpolation version. Double-check using brute-force original models!"""

    import sys
    import h5py
    import numpy as N

    sys.path.append('/home/robert/science/sedfit')
    import ndinterpolation

    ip = ndinterpolation.LocIntp(clumpydb,type='torus',order=1,mode='log')

    hc = h5py.File(clumpydb,'r')
    i = hc['i'][:]
    co0 = (i == 0)
    sig = hc['sig'][co0]
    N0 = hc['N0'][co0]
    tv = hc['tv'][co0]
    Y = hc['Y'][co0]
    q = hc['q'][co0]
    hc.close()
#    i = i[co0]
    print "sig.size = ", sig.size


    F0 = N.zeros((sig.size,len(filterwaves)))
    F90 = N.zeros((sig.size,len(filterwaves)))

    for j in range(sig.size):
        if j % 1000 == 0:
            print j, ' of ', sig.size
 
        params0 = (0.,tv[j],q[j],N0[j],sig[j],Y[j])
        params90 = (90.,tv[j],q[j],N0[j],sig[j],Y[j])
        for k,w in enumerate(filterwaves):
            w_ = N.array(w)
            F0[j,k] = ip(params0,w_)
            F90[j,k] = ip(params90,w_)

    return F0, F90

#
#
##    fluxes0 = fluxes[co0,:]
##    fluxes90 = fluxes[co90,:]
#
#    print "fluxes.shape, fluxes0.shape, fluxes90.shape = ", fluxes.shape, fluxes0.shape, fluxes90.shape
#
#    del fluxes, fluxes0, fluxes90, i
    



def get_clumpy_fluxratios_classic(clumpydb='/fast/robert/data/clumpy/models/20120609_tvavg.hdf5',filterwaves=(3.4,4.6,12.),from_npzfile=None,save_npzfile=None):

    """This is the ndinterpolation version. Double-check using brute-force original models!

    Currently standard values for from_npzfile & save_npzfile:
       from_npzfile = /home/robert/science/wise_colors/type2_normalization/F0_F90.npz
       save_npzfile = /home/robert/science/wise_colors/type2_normalization/F0_F90.npz

    """

    import sys
    import h5py
    import numpy as N
    from scipy import interpolate

    if from_npzfile is not None:
        print "Loading F0 & F90 from npz file."
        npzfile = N.load(from_npzfile)
        F0, F90 = [npzfile[e] for e in ['F0','F90']]
        npzfile.close()
        return F0, F90

    else:
        sys.path.append('/home/robert/science/sedfit')
        import ndinterpolation

        hc = h5py.File(clumpydb,'r')
        wave = hc['wave'][:]
        i = hc['i'][:]
        co0 = (i == 0)
        co90 = (i == 90)
        print "co0[N.where(co0==True)].size = ", co0[N.where(co0==True)].size
        print "co90[N.where(co90==True)].size = ", co90[N.where(co90==True)].size

        idx = N.arange(i.size)

        idx0 = idx[co0]
        idx90 = idx[co90]

        fluxes = hc['flux_tor'][:,:]

        hc.close()

        F0 = N.zeros((idx0.size,len(filterwaves)))
        F90 = N.zeros((idx90.size,len(filterwaves)))

        for j in range(idx0.size):
            if j % 1000 == 0:
                print j, ' of ', idx0.size

            ip0 = interpolate.interp1d(wave,fluxes[idx0[j]])
            ip90 = interpolate.interp1d(wave,fluxes[idx90[j]])

            for k,w in enumerate(filterwaves):
#                w_ = N.array(w)
                F0[j,k] = ip0(w)
                F90[j,k] = ip90(w)


    if save_npzfile is not None:
        print "Saving F0 & F90 to npz file %s" % str(save_npzfile)
        N.savez(save_npzfile,F0=F0,F90=F90)

        
    return F0, F90



########### OPTIMIZE #############

def opt():


    # 1) Load somehow defined blue rectangle data (clean)

    # 2) Red rectangle. Find optimum maximal area for the red rectangle.

    """
    We'll be trying to find some 'optimal' countour line in the
    color-color diagram.

    Each time the countour line is modified, two histograms are
    calculated: the number count per W1 bin of measured T2 sources,
    and the number count per W1 bin of predicted T2 sources.

    At each such step, compare the two histograms to each other. The
    comparison measure could e.g. be an avg. (absolute) deviation per
    histogram bin, or a KLD value (since the histograms are
    distributions of counted source numbers per magnitude bin).

    Optimize the countour line such that the encompassed area in the
    CC diagram is as large as possiible (and contiguous), w/o
    exceeding some critical value of the measure/cost function.

    y(x) = 

    """


def generate():

    ug = pymc.Uniform('u',low,up)

    for j in range(n):
#        if j % 100 == 0: print "Model %d of %d" % (j,n)
        theta = ug.rand()
        print "Model %d of %d, theta = " % (j,n), theta
        yield theta





#oldgooddef workerfunc(modelnumber,theta,**kwargs):
#oldgood    print "[%02d] modelnumber: " % agent.rank, modelnumber
#oldgood
#oldgood    t1box = kwargs['t1box']
#oldgood    t2box = kwargs['t2box']
#oldgood
#oldgood    r_ = theta
#oldgood    print "modelnumber, theta, type(theta) = ", modelnumber, theta, type(theta)
#oldgood
#oldgood    # do something with this theta-vector of parameter values
#oldgood    v = N.random.uniform(0.,1.,size=m)  # viewing; cos(i) random number generator; uniform, 1-dimensional
#oldgood#    print "v, type(v) = ", v, type(v)
#oldgood
#oldgood    seds = N.zeros((m,ip.wave.size))
#oldgood    for jv in range(m):
#oldgood#            seds[jv,:] = ip(N.concatenate(([v[jv]],r_)),ip.wave)
#oldgood        seds[jv,:] = ip(N.concatenate(([1.0-v[jv]],r_)),ip.wave)  # See the CAUTION under 2)!!!
#oldgood
#oldgood    # 5)
#oldgood    # For all SEDs for this model, calculate filterfluxes in W1--W3, and colors W1-W2 & W2-W3
#oldgood    filterfluxes = get_filterfluxes(filters, vega_normalizations,ip.wave,seds)
#oldgood    colors = get_colors(filterfluxes)  # CAUTION: colors.shape = (n,2), and colors[:,0] = W1-W2, colors[:,1] = W2-W3
#oldgood
#oldgood    # 6)
#oldgood    # For this model, find all viewings which have colors in type1-region ('blue box'), and simultaneously in type2-region ('red box')
#oldgood    t1_indices_true, t1_indices_false = points_in_polygon(colors,t1box)
#oldgood    if t1_indices_true.size == 0:
#oldgood        return [-1]   #continue  # skip this model if no viewings with colors in t1-box found
#oldgood
#oldgood    t2_indices_true, t2_indices_false = points_in_polygon(colors,t2box)
#oldgood    if t2_indices_true.size == 0:
#oldgood        return [-1]   #continue  # skip this model if no viewings with colors in t2-box found
#oldgood
#oldgood    # we only get here if both t1_indices_true and t2_indices_true aren't empty
#oldgood    filterfluxes_t1 = filterfluxes[t1_indices_true,:]
#oldgood    filterfluxes_t2 = filterfluxes[t2_indices_true,:]
#oldgood
#oldgood    filterfluxes_t1_mean = filterfluxes_t1.mean(axis=0)  # mean model flux for all viewings with compatible t1 colors
#oldgood    filterfluxes_t2_mean = filterfluxes_t2.mean(axis=0)  # mean model flux for all viewings with compatible t2 colors
#oldgood
#oldgood    Dm = 2.5*N.log10(filterfluxes_t1_mean/filterfluxes_t2_mean)  # 1-D array of 3 numbers (this is for the current model, all viewings involved)
#oldgood#    print "WORKERFUNC: Dm = ", Dm
#oldgood
#oldgood#    return [theta, Dm]
#oldgood    return Dm, theta, v[t1_indices_true].mean(), v[t2_indices_true].mean()
#oldgood#    return None #[subseq, 5.]


def workerfunc(modelnumber,theta,**kwargs):

    """
    0) Get parameter vector.
    1) Generate m random viewings.
    2) Interpolate m SEDs.
    3) For each SED, calculate 4 filter fluxes.
    """

    # 1)
    v = N.random.uniform(0.,1.,size=m)  # viewing; cos(i) random number generator; uniform, 1-dimensional

    # 2)
    seds = N.zeros((m,ip.wave.size))
    for jv in range(m):
        seds[jv,:] = ip(N.concatenate(([1.0-v[jv]],theta)),ip.wave)  # See the CAUTION under 2)!!!

    # 3) For all SEDs for this model, calculate filterfluxes in W1--W3
    filterfluxes = get_filterfluxes(filters,vega_normalizations,ip.wave,seds)

    # 4) colors
    colors = get_colors(filterfluxes)  # CAUTION: colors.shape = (n,3), and colors[:,0] = W1-W2, colors[:,1] = W2-W3, colors[:,2] = W3-W4

    return theta, v, filterfluxes, colors


#listdef resultfunc(idx,li,a,**kwargs):
#list
#list#    print "RESULTFUNC: li, a: ", li, a
#list#    print "RESULTFUNC: li, a: ", li, a
#list
#list    if li is None:
#list        li = []
#list
#list    if isinstance(a[0],int):
#list        pass
#list    else:
#list        li.append(a[0])
#list
#list    return li


#dictdef resultfunc(idx,di,a,**kwargs):
#dict
#dict    if di is None:
#dict        di = {'Dm' : [], 'theta' : [], 'v1' : [], 'v2' : []}
#dict
#dict    if isinstance(a[0],int):
#dict        pass
#dict    else:
#dict        di['Dm'].append(a[0])
#dict        di['theta'].append(a[1])
#dict        di['v1'].append(a[2])
#dict        di['v2'].append(a[3])
#dict
#dict    return di


def resultfunc(idx,di,a,**kwargs):
    
    print "RESFUNC: idx = ", idx
    
    ds_theta[idx-1,:] = a[0]
    ds_cosi[idx-1,:] = a[1]
    ds_filterfluxes[idx-1,:,:] = a[2]
    ds_colors[idx-1,:,:] = a[3]


def setup(dbfile):

#root needs: n, m,  low, up, ug
#ranks need: n, m,  ip, filters, vega_normalizations

#    n = int(1.5e5)   # number of random models
    n = 100 # TESTING
    m = 100    # number of random viewings per model

    # setups for all ranks
    # 0)
    if agent.rank != 0:
        print "[%02d] Instantiating SED interpolator object..." % agent.rank
#        ip = ndinterpolation.LocIntp(dbfile,type='torus',order=1,mode='log')
        ip = ndinterpolation.LocIntp(dbfile,type='torus+bb',order=1,mode='log')
        low = [par[0] for par in ip.theta[1:-1]]
        up  = [par[-1] for par in ip.theta[1:-1]]
        filters, vega_normalizations = get_filters()
        hdo, ds_theta, ds_cosi, ds_filterfluxes, ds_colors = None, None, None, None, None
    else:
        ip, low, up = None, None, None
        print "[%02d] Creating HDF5 output file..." % agent.rank
        filters, vega_normalizations = None, None
#R        hdo = h5py.File('./type2norm_20130424.hdf5','w')
        hdo = h5py.File('./type2norm_20130629_torus+bb.hdf5','w')
        # HDF5 datasets
        ds_theta = hdo.create_dataset('theta',shape=(n,5),dtype=N.float32)
        ds_cosi = hdo.create_dataset('cosi',shape=(n,m),dtype=N.float32)
        ds_filterfluxes = hdo.create_dataset('filterfluxes',shape=(n,m,4),dtype=N.float32)
        ds_colors = hdo.create_dataset('colors',shape=(n,m,3),dtype=N.float32)

#    if agent.rank >= 0: print "[%02d] BEFORE: low, up = " % agent.rank, low, up
    low = agent.bcast(low,source=1)
    up = agent.bcast(up,source=1)
#    if agent.rank >= 0: print "[%02d] AFTER: low, up = " % agent.rank, low, up

    return n, m, ip, low, up, filters, vega_normalizations, hdo, ds_theta, ds_cosi, ds_filterfluxes, ds_colors



if __name__ == '__main__':

    import sys
    import pymc
    sys.path.append('/home/robert/science/sedfit')
    import ndinterpolation
    from communication import ServerClient

    # constants
    dbfile = '/fast/robert/data/clumpy/models/20130415_tvavg_resampled.hdf5'

    # instantiate ServerClient
    agent = ServerClient()

    # root do some setup
    n, m, ip, low, up, filters, vega_normalizations, hdo, ds_theta, ds_cosi, ds_filterfluxes, ds_colors = setup(dbfile)

    # run task
#    agent(generate(n,low,up), workerfunc, returnfunc)
#    agent(generate(n,low,up), workerfunc, None)

#good    agent(generate(), workerfunc, resultfunc, t1box=(1.,1.5,2.5,3.5), t2box=(2.,3.,4.,5.))
    agent(generate(), workerfunc, resultfunc)

    if agent.rank == 0:
        hdo.close()
##        print "type(agent.results), len(agent.results), agent.results[0].size, agent.results[1].size  : ", type(agent.results), len(agent.results), agent.results[0].size, agent.results[1].size
#        fi = open('./Dm_mpi_1e6_100.npz','w')
#        N.savez(fi,theta=N.array(agent.results['theta']),v=N.array(agent.results['v']),filterfluxes=N.array(agent.results['filterfluxes']))
#        fi.close()
    
    agent.mpifinalize()
