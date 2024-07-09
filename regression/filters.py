import os, h5py
import numpy as N
from scipy import integrate

"""Module for handling photometry filters.

class FilterLib

  __init__ 
    Instantiates a filter library class from an HDF file, or creates a
    new one.

  __call__
    If all *args are Filter() instances, stores the filters into the
    lib. If all *args are strings, retrieves from the lib all filters
    with the names given by *args, and returns them as a list of
    Filter() instances.


Ideas/TODOs:

+ Import one filter per file, or whole directory.
(+) Allow addition and removal of filters to the lib.
- For additon, allow import from ASCII files, FITS files (tables?)
(+) Allow export of single filters to: txt, fits.
- Allow export of whole lib to: one HDF5 file, one FITS file, one ASCII file, set of ASCII files.
- write FilterLib.delete() function; uses FilterLib._delete()
- Have methods to average a supplied spectrum with any filter (or multiple filters)
- Clean up / remove old code.

"""

__author__ = 'Robert Nikutta <robert@pa.uky.edu>'
__version__ = '2024-07-08'  # yyyy-mm-dd
__version_first__ = '2011-02-09'


#! ########### PLAYGROUND ##############
#! 
#! class Filter
#! class FilterLib
#! 
#! lib = FilterLib()
#! # FilterLib's __call__:
#! lib(filter)       # --> store filter in lib
#! lib(filtername)   # --> retrieve filter from lib
#! 
#! filternames = ['fname1','fname2','fname3']
#! filters = [lib[fname] for fname in filternames]  # list of Filter() instances
#! 
#! # how to apply all filters in 'filters' to same model
#! f = filters[0]
#! f(modelwave,modelflux) # the filter's __call__ method returns: clam, modelflux_after_filter
#! 
#! # Do it for every requested filter --> lib has convenience function
#! # for it, and returns two 1D arrays: clam[:] and modelflux_after_filter[:]
    

# /////////////////////////////////////////////////////////
# CLASSES
# /////////////////////////////////////////////////////////

# /////////////////////////////////////
class Filter:
    """Single filter class.

    Brainstorming

    - must be able to feed filter data directly, or via FilterLib() class from a library.
    - if feeding data in directly:
      - lam, phi(lam), what normalization is this? (determine automatically?)
    

    """

    def __init__(self,name,lam,phi,clam=None,normalization_input='raw',norm=1.,normalization_use='raw',lambdafactor=1.):
        """

        Mandatory arguments:

          name, lam, phi

        Optional arguments:
        
          clam = None or float   --> if None, phi.clam will be calculated
                                 --> if float, phi.clam will be set to clam

          normalization_input = raw / area / peak   --> if raw, norm=1. (even if specified otherwise), and phi_raw = phi * norm = phi * 1.
                                                    --> if area, must also specify norm (i.e. \int phi_raw d\lam). Then phi_raw = phi * norm
                                                    --> if peak, must also specify norm (i.e. phi_raw.max()). Then phi_raw = phi * norm

          normalization_use : str
            raw / area / peak
              --> always store phi_raw and normalization_use; then
                  invoking Filter.phi (a function!) will always return a
                  properly normalized function
        """

        try:
            self.name = str(name)
        except:
            raise

        self.lam = lam
        phi_ = phi

        # scale the wavelengths by requested factor (e.g. 1.e-4 to go from micron to Angstrom)
        # phi(lam) will be scaled opposite such that \int dlam phi(lam) = 1.
        self.lam *= lambdafactor
        phi_ /= lambdafactor

        # if provided, store clam; else calculate and store clam
        self.clam = clam if clam else integrate.trapz(self.lam*normalize(self.lam,phi_),self.lam)

        # always store phi_raw, after applying normalization
        if normalization_input == 'raw':
            self.norm = 1.
        elif normalization_input in ['area','peak']:
            self.norm = norm
        else:
            raise ValueError("Invalid value for 'normalization_input'. Chose from raw/area/peak.")

        self.phi_raw = phi_ * self.norm


        # TODO: try just calling self.set_normalization instead of the following two lines
        self.normalization_use = normalization_use
        self.phi = self.calculate_phi()


    # /////////////////////////////////////
    def calculate_phi(self):
        """From phi_raw(lam) calculate phi(lam), normalized as requested by 'normalization_use'."""
        if self.normalization_use == 'raw':
            norm = 1.
        elif self.normalization_use == 'area':
            norm = 1. / integrate.trapz(self.phi_raw,self.lam)
        elif self.normalization_use == 'peak':
            norm = 1. / self.phi_raw.max()

        return self.phi_raw * norm
    # /////////////////////////////////////


    # /////////////////////////////////////
    def set_normalization(self,newnorm):
        self.normalization_use = newnorm
        self.phi = self.calculate_phi()   # trigger re-calculation of phi(lam) according to requested normalization
    # /////////////////////////////////////



#1    # /////////////////////////////////////
#1    def __init__(self,name,normalization,lam,phi,clam=None):
#1        """Instatiate single filter.
#1
#1        Filter function phi(lam) will always be normalized to unit area.
#1
#1        Arguments
#1        ---------
#1        name : str
#1          self.name is the filter name, e.g. 'iracH'. Must not be empty.
#1
#1        lam : array
#1          Wavelengths in micron.
#1
#1        phi : array
#1          Filter function phi(lam). Will be normalized to unit area.
#1
#1        clam : float, or length-one float array, or None (default)
#1          Central wavelength of filter function. If None, will be calculated (see source below).
#1
#1
#1        Class attributes
#1        ----------------
#1        self.name : str
#1          As argument 'name'. Empty string '' if name==None.
#1
#1        self.lam : array
#1          As argument 'lam'.
#1
#1        self.phi : array
#1          As argument 'phi', normalized to unit area, i.e. \int dlam phi(lam) = 1.
#1
#1        self.clam : length-one float array
#1          Central wavelength of the filter, defined as \int dlam lam*phi(lam)
#1
#1        # Scratch area
#1        Get data from where?
#1          - A library (HDF5 file) ?
#1          - A text file?
#1        Store library how?
#1          - HDF5 file?
#1          - Text file?
#1        """
#1
#1        if isinstance(name,str):
#1            self.name = name
#1        else:
#1            raise Exception, "No valid name provided for filter. Name must be a string."
#1
#1        self.lam = lam
#1
#1        # temporarily normalize phi to unit area for calculating clam
#1#        self.phi = normalize(self.lam,phi)
#1
#1        # if provided, store clam; else calculate and store clam
#1#        self.clam = clam if clam else integrate.simps(self.lam*self.phi,self.lam)
#1        self.clam = clam if clam else integrate.simps(self.lam*normalize(self.lam,phi),self.lam)
#1        self.clam = np.array([self.clam])  # turn into length-one array; if clam was an array already, this doesn't do anything
#1
#1        # store un-normalized, raw phi (FilterLib.get() function will provide normalized phi functions)
#1        self.phi = phi
#1
#1        self.normalization = normalization
#1        


    def __call__(self):
        """TODO: Calling the filter with some (x,y) model should apply
        the filter to the model, and return the central wavelength of
        the filter, and the flux of the model after applying the
        filter."""
        pass


# ////////// end class Filter //////////


# /////////////////////////////////////
class FilterLib:
    """Class to handle a filter library."""

    # /////////////////////////////////////
    def __init__(self,hdffile):
        """Initialize HDF filter library.

        If 'hdf' file does not exist, create it. If it does exist,
        open it for reading and appending. The names of filters stored
        in the HDF file are case-sensitive, but when adding new
        filters, the namespace is checked for object with the same
        name ignoring the case.
        """

        # allowed modes of normalization
        self.normalizations = ['raw','area','peak']

        # this opens an existing HDF file for read/write access;
        # creates the file if it does not exist yet
        self.file = os.path.realpath(hdffile)
        try:
            self.lib = h5py.File(self.file,'a')
        except:
            raise Exception("Filter library file '%s' not found or invalid." % self.file)


    def close(self):
        """Close the underlying HDF5 of the opened filter library object."""

        self.lib.close()


    def __call__(self,*args,**kwargs):
        """Return from or store to lib an instance of Filter() class.

        Description
        -----------
        If all *args are instances of Filter class, the data of these
        filters will be stored in the HDF library. If all *args are
        instead strings, they will be considered names of filters
        already present in the library, and a list of Filter class
        instances for these filters will be returned.

        Examples
        --------
        Retrieve two filters from library:

          lib = FilterLib(hdffile)   # hdffile is path to the library file
          lib.filternames()          # may return ['F222M','F606W','timmi2K']
          filters = lib('F222W','timmi2K')
          for f in filters: print(f,f.name)
          --> 
             <filters.Filter instance at 0x34acf38> timmi2K
             <filters.Filter instance at 0x34d2248> F222M
        """

        # return, if args empty
        if not args:
            return

        # put all *args to a list
        args = flat(args)
        
        if allinstance(args,Filter):
            res = self.store(args,kwargs)   # returns None if all went fine
        elif allinstance(args,str) or allinstance(args,unicode):
            normalization = kwargs.get('normalization','area')   # set default normalization to 'area', if no value was provided
            res = self.get(args,normalization)                   # returns a list of Fliter() class instances
        else:
            raise Exception("Supplied argument(s) are neither all strings nor all instances of Filter class.")

        return res


    def _delete(self,filtername):
        """Delete filter from the library.

        This function should be considered non-public. It also is
        aggressive (no safety checks), so do your checks outside.
        """

        # TODO: this sanity check will go once FilterLib.delete() func defined
        if not isinstance(filtername,str):
            raise Exception("Supplied filter name is not a string.")

        try:
            del self.lib[filtername]
        except KeyError:
            raise Exception("Filter name '%s' is not present in the library." % filtername)


    def _get(self,filtername,normalization='area'):
        """Return instance of class Filter() for a single filter name.

        This function should be considered private and not member of
        the public API. Use function get_filters() publicly. Also,
        perform all sanity checks outside; this function is agressive
        and raw.

        The 'filtername' must be of a filter present in the HDF library.

        Parameters
        ----------
        filtername : str
            Name of single filter that is present in the HDF library.

        normalization: str
            Must be one of the following (default is 'area')
              'none' = raw phi(lambda) function (as provided by original filter specification file)
              'area' = phi(lambda) normalized to unit area, i.e. \int phi(lambda) dlambda = 1
              'peak' = phi(lambda) = 1 at peak value
        """

        # --- fetch filter from HDF
        clam_ = self.lib[filtername + '/clam'][:]
        lam_  = self.lib[filtername + '/lam'][:]
        phi_  = self.lib[filtername + '/phi'][:]  # this is the raw phi(lambda)

        # normalize phhi(lambda) if requested
        if normalization == 'area':
            phi_ = normalize(lam_,phi_)
        elif normalization == 'raw':
            pass
        elif normalization == 'peak':
            phi_ = phi_ / phi_.max()
        else:
            raise ValueError("Invalid value for 'normalization'! Allowed: none/area/peak. Got: %s." % normalization)

        print("clam_",clam_)
        return Filter(filtername,lam_,phi_,clam=clam_,normalization_input=normalization)


    def get(self,filternames,normalization='area',sortkey='clam'):
        """Return instances of class Filter() for all requested filter names.

        This function is part of the public API.

        All requested filternames must be of a filters present in the
        library, otherwise an exception will be raised.

        Parameters
        ----------
        filternames : list of strings
            Names of filter names. All names must be present in the
            HDF library.

        sorting : str
            ['clam', 'name']
        """

        if normalization not in self.normalizations:
            raise Exception("Invalid normalization '%s'. Must be one of: '%s'." % (normalization,'/'.join(self.normalizations)))
        else:
            print("Using filter normalization '%s'." % normalization)

        # turn filternames into list, if it is not; then into a set
        filternames = set(sequentialize(filternames))

        # --- check if supplied filter is an instance of the Filter() class
        if not allinstance(filternames, str):
            raise Exception("Some supplied filternames are not strings.")

        # check if all requested filters are in the lib
        intersection = set.intersection(filternames,set(self.filternames()))
        if sorted(intersection) != sorted(filternames):
            raise Exception("Not all requested filters are present in the library. Availables filters are: %s" % ','.join(self.filternames()))

        # Return list of filters; even if only one filter names was
        # requested, a list (of length one) will be returned. This is
        # the principle of least surprise. The return value is always
        # a list.
        filterlist = [self._get(fname,normalization=normalization) for fname in filternames]

        if sortkey == 'name':
            func = lambda f: f.name.lower()
        elif sortkey == 'clam':
            func = lambda f: f.clam

        return sorted(filterlist,key=func)


    def filternames(self):
        """Return list of filter names present in the library."""

        return sorted([str(f) for f in self.lib.keys()], key=lambda x: x.lower())


    def _store(self,filter):
        """Take single Filter() class instance and store its data into library.

        This function should be considered private and not member of
        the public API. Use public function store() instead.

        This is an agressive function: if the filter already exists in
        the lib, it will be overwritten without safety check. Do your
        safety checks outside of this function!

        It also does not check for consistency of input. Check it
        outside.

        Parameters
        ----------
        filter : single instance of Filter() class
            Instance of Filter() class to be stored in HDF library.
        """

        # --- delete filter from lib if it already exists there
        if filter.name in self.filternames():
            self._delete(filter.name)

        # --- store filter to HDF lib
        # create a group for this filter
        group = self.lib.create_group(filter.name)

        # store numbers in to data sets, creating on the fly data sets within group
        lam_  = group.create_dataset('lam', data=filter.lam)    # lambda values
        phi_  = group.create_dataset('phi', data=filter.phi)    # phi transmission curve
        clam_ = group.create_dataset('clam',data=filter.clam)   # central wavelength of the filter

        return None


    def store(self,filters,overwrite=False):
        """Store data from supplied Filter() class instances into HDF library.

        This is a public convenience function. Performs all checks.

        Parameters
        ----------
        library : instance of FilterLib class
            A FilterLib instance into which to store the supplied filters.

        filters : list of instances or single instance
            List of instances of Filter() class. If single instance,
            will be turned into a list.

        overwrite : bool
            If True, allow overwriting of filters already present in
            the lib. If False and at least one of the filters present
            in the lib, report that, do nothing else, and return
            undone.
        """

        # turn filters into list, if it is not
        filters = sequentialize(filters)

        # make a set of filter names
        filternames = set([f.name for f in filters])
        
        # ensure that no two supplied filters have the same name
        if len(filters) != len(filternames):
            raise Exception("Some supplied filters have identical names.")

        # check if all elements are instances of Filter class
        if not allinstance(filters,Filter):
            raise Exception("Some elements in list are not instances of Filter class.")

        # any filter names already in library?
        intersection = set.intersection(filternames,set(self.filternames()))
        if intersection:   # if yes...
            msg = "Filter(s) '%s' is/are already present in the library. " % ",".join(intersection)
            if not overwrite:
                raise Exception(msg + "You can overwrite via overwrite=True.")
            else:
                print(msg + "Will be overwritten.")

        # now store all filters in lib
        for f in filters:
            self._store(f)

# ////////// end class FilterLib //////////




# /////////////////////////////////////////////////////////
# MODULE CONVENIENCE FUNCTIONS
# /////////////////////////////////////////////////////////

def load_filters_from_directory(path,suffix='.dat',lambdafactor=1.):

    """Read all filter files (ascii) in directory 'path'.

    Turn all files into instances of Filter() class, return a list of
    instances.

    """

    assert (os.path.isdir(path))

    path = os.path.normpath(path)

    files = os.listdir(path)
    files = sorted([f for f in files if f.endswith(suffix)],key=lambda s: s.lower())
    filternames = [f[:f.find(suffix)] for f in files]

    filters = [] #Filter(f,clam=None
    for j,f in enumerate(files):
        filt = read_filter_file(path+os.path.sep+f,skiprows=1,suffix=suffix,normalization_input='raw',norm=1.,normalization_use='area',lambdafactor=lambdafactor)
        filters.append(filt)
#        lam, phi = N.loadtxt(path+os.path.sep+f,usecols=(0,1),skiprows=1,unpack=True)   # read in the data
#        filters.append(Filter(filternames[j],lam,phi,clam=None,normalization_input='raw',norm=1.,normalization_use='area'))

    return path, filters


def read_filter_file(filepath,skiprows=1,suffix='.dat',normalization_input='raw',norm=1.,normalization_use='area',lambdafactor=1.):
    """Read single-filter data from a text file.

    Description
    -----------
    The format of the text file must be 2-column (from left to right):

      1. Filter wavelength lambda in micron, in ascending order.

      2. Filter function phi(lambda). This will be normalized to
      either unit area, peak value, or left as-is ('raw')

    Any line beginning with # will is considered a comment line, and
    will be ignored. If not zero, the first 'skiprows' rows will not
    be read; default is skiprows=1.

    The filter name will be the final part of the full path, minus the
    file suffix.

    The function also understands keywords that deal witht he
    normalization of the transmission curve.

    Parameters
    ----------
    filepath : str
        Path to text file to be read.

    skiprows : int
       Number of first rows to be skipped while reading. Default: 0.

    normalization_input, norm, normalization_use
        For all these, please see the docstring of Filter() class.

    Return
    ------
    Instance of class Filter() (see docstring there).

    """

    path = os.path.normpath(filepath)   # normalize path

    if not os.path.isfile(filepath):   # is path a proper file?
        raise Exception("File path %s is not a regular file." % filepath)

    basename = os.path.basename(filepath)
    filtername = basename[:basename.find(suffix)]
    lam, phi = N.loadtxt(filepath,usecols=(0,1),skiprows=skiprows,unpack=True)   # read in the data

    # instantiate class Filter() with the read-in data
    myfilter = Filter(filtername,lam,phi,clam=None,normalization_input=normalization_input,norm=norm,normalization_use=normalization_use,lambdafactor=lambdafactor)

    # return the instance
    return myfilter


def flat(li):
    """Simple function to flatten a sequence. Works recursively.
       types=(list)        flattens lists until first non-list level
       types=(tuple)       flattens tuples until first non-tuple level
       types=(list,tuple)  flattens lists and tuples
    """

    newli = []
    for l in li:
      if isinstance(l,(list,tuple,set)):
        newli += flat(l)
      else:
        newli += [l]

    return newli


def sequentialize(obj):
    """If object not list or tuple, return it in a list."""

    return obj if isinstance(obj,(list,tuple)) else [obj]


def normalize(lam,phi):

    """Normalize phi(lam) to unit area, i.e. \int dlam phi(lam) = 1."""

    return phi / integrate.trapz(phi,lam)


def allinstance(seq,class_):
    """Check if all elements in sequence are instances of class_."""

    return True if all( [isinstance(elem,class_) for elem in seq] ) else False



#3# /////////////////////////////////////
#3def delete(library,filternames):
#3    """Convenience function to delete one or more filters from filter library.
#3
#3    'filternames' is list of strings (or will be turned into one),
#3    with the strings being names of filters stored in 'library'.
#3
#3    Check all items for validity before actually deleting.
#3
#3    This function is considered part of the public API.
#3    """
#3
#3    # is library instance of class FilterLib?
#3    if not isinstance(library,FilterLib):
#3        raise Exception, "Library object is not instance of FilterLib class."
#3
#3    # make filternames a list, if necessary
#3    filternames = sequentialize(filternames)
#3
#3    # are all names actually strings?
#3    if not ([isinstance(fn,str) for fn in filternames]).all():
        





######################### OLD STUFF

#    # /////////////////////////////////////
#    def get_filter(self,filternames):
#        """Return from the library an instance of class Filter() for requested filtername."""
#
#        filternames = sequentialize(filternames)
#        
#        for name in filternames:
#            if name in self.filternames:
#                lam = self.lib[name + '/lam']
#                phi = self.lib[name + '/phi']
#                cwave = self.lib[name + '/clam']
#                filt = Filter(lam,phi,name,norm=False)
#                self.openfilters.append(filt)
#    # /////////////////////////////////////






#def get_filters(lines,hdffilterfile):
#    import h5py
#
#    hdf = h5py.File(hdffilterfile,'r')
#    hdf_filters = hdf.keys()
#
#    file_filters = []
#    for line in lines:
#        elems = line.strip().split()
#        myfilter = elems[0].strip().replace("'","")
#        if myfilter in hdf_filters:
#            file_filters.append(myfilter)
#
#    hdf.close()
#    return hdf_filters, file_filters


#def list_filters(f,sortedby='filtername'):
#    """List all filters and their central wavelengths available in given HDF5 file f.
#
#    Arguments:
#
#    sortedby = 'filtername' / 'centralwave'
#       Output table sorted either by the filter name (default) or by central wavelength.
#
#    """
#
#    import h5py
#
#    h = h5py.File(f,'r')
#
#    filternames = h.keys()
#    central_waves = []
#    number_points = []
#
#
#    for k in filternames:
#        central_waves.append( h[k]['central_wave'][0] )
#        number_points.append( h[k]['wave'].shape[0] )
#
#    if sortedby == 'filtername':
#        idx = (0,1,2)
#    elif sortedby == 'centralwave':
#        idx = (1,0,2)
#
#    elems = (filternames,central_waves,number_points)
#    sortedlist = sorted(zip(elems[idx[0]],elems[idx[1]],elems[idx[2]]))
#    filters_      = [j[idx[0]] for j in sortedlist]
#    centralwaves_ = [j[idx[1]] for j in sortedlist]
#    numberpoints_ = [j[idx[2]] for j in sortedlist]
#        
#    print "Available filters in HDF5 file %s" % f
#    print "-----------------------------------------------------"
#    print "         filter    central wavelength    number of"
#    print "           name        [micron]          data points"
#    print "-----------------------------------------------------"
#    for j in range(len(filters_)):
#        print "%15s     %10.5f           %d" % (filters_[j],centralwaves_[j],numberpoints_[j])
#
#    return filters_, centralwaves_
    

#def ascii2hdf(d,normalize='area',skiprows=1,hdffile=None,verbose=0):
#    """Turn a set of ASCII files of filter functions into one HDF5 file.
#
#    Have only ASCII files of filter transmission curves in the
#    directory d.  One file per filter. The file's name will be the
#    filter name in tye HDF5 file (up to the first dot, if any).  All
#    transmission curves will be normalized to unit area (as a function
#    of wavelength) by default.  Each ASCII file must contain 2
#    columns: (1) wavelength in micron (2) filter transmission.  The
#    number of (non-comment) lines to be skipped before the filter
#    curve begins can be controlled by 'skiprows' (default is 1).  The
#    name of the resulting HDF5 file must be given as 'hdffile'.  If
#    verbose > 0 (the default), output progress messages.
#
#    Structure of HDF5 file:
#
#    /
#    /filter1
#    /filter1/wave
#    /filter1/transmission
#    /filter1/central_wave
#    /filter2
#    /filter2/wave
#    /filter3/transmission
#    /filter4/central_wave
#    ...
#
#    /filterX are groups
#
#    /filterX/wave is a dataset (array of shape (nwave,)) holding the
#    wavelengths (in micron)
#
#    /filterX/transmission is a dataset of same shape as wave, holding
#    the filter transmission curve (normalized to unit area)
#
#    /filterX/transmission is a dataset (shape (1,)) that holds the
#    central wavelength of that filter (defined as
#      \int dwave wave*transmission
#
#    """
#
#    import os, sys, h5py
#    import numpy as np
#    from scipy import integrate
#
#    root = os.path.normpath(d) + os.path.sep
#    ascii_files = sorted( os.listdir(d) )
#
#    h = h5py.File(hdffile,'w')
#
#    counter = 0
#    for f in ascii_files:
#        counter += 1
#        number_of_files = len(ascii_files)
#        if verbose > 0:
#            print "Working on file %d / %d : %s" % (counter,number_of_files,f)
#            sys.stdout.flush()
#        
#        name = f.split('.')[0]
#
#        # load wave and transmission curve, calculate normalized curve & central wavelength
#        wave, transmission = np.loadtxt(root + f,unpack=True,usecols=(0,1),skiprows=skiprows)
#        transmission = transmission / integrate.simps(transmission,wave)
#        central_wave = integrate.simps(wave*transmission,wave)
#
#        # create one group per filter
#        group = h.create_group(name)
#
#        # create data sets withing group
#        wave_ds         = group.create_dataset('wave',(wave.size,),'float64')
#        transmission_ds = group.create_dataset('transmission',(transmission.size,),'float64')
#        central_wave_ds = group.create_dataset('central_wave',(1,),'float64')
#
#        # store numbers in to data sets
#        wave_ds[:] = wave[:]
#        transmission_ds[:] = transmission[:]
#        central_wave_ds[:] = central_wave
#
#    h.close()
    
