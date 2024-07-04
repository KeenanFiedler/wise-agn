import numpy as np
import h5py as h5
import math
import sys
np.set_printoptions(threshold=sys.maxsize)

infile = h5.File("C:\\Users\\keena\\Documents\\University of Arizona\\Jobs\\TIMESTEP NOIRLAB\\wise-agn\\pca\\clumpy_models_201410_tvavg.hdf5", 'r')
print("Keys: %s" % infile.keys())
#get the correct data
Y = infile['Y'][:]
ind = np.where(Y==5)
Y = Y[ind]
N0 = infile['N0'][ind]
flux_tor = infile['flux_tor'][ind]
i = infile['i'][ind]
q = infile['q'][ind]
sig = infile['sig'][ind]
tv = infile['tv'][ind]
wave = infile['wave']
print(N0[:1000])
#N0 = infile['N0'][:]
#print("Keys: %s" % dset.keys())
file_dtype = [('N0', N0.dtype),
              ('Y', Y.dtype),
              ('flux_tor', flux_tor.dtype, len(wave)),
              ('i', i.dtype),
              ('q', q.dtype),
              ('sig', sig.dtype),
              ('tv', tv.dtype),
              ('wave', wave.dtype, len(wave))]
out_array = np.zeros((len(Y),),dtype=file_dtype)
out_array['N0'] = N0
out_array['Y'] = Y
out_array['flux_tor'] = flux_tor
out_array['i'] = i
out_array['q'] = q
out_array['sig'] = sig
out_array['tv'] = tv
out_array['wave'] = wave
infile.close()


outfile = h5.File("C:\\Users\\keena\\Documents\\University of Arizona\\Jobs\\TIMESTEP NOIRLAB\\pca\\models_subset.hdf5", 'w') #open file called test_config.hdf5 in write mode (i.e. 'w')
outfile.create_dataset("dset", (len(Y),), dtype = file_dtype, data = out_array, chunks=True, compression = "gzip") #create a new dataset called dset of given size and shape and fill it with out_array
outfile.flush()