import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import healpy as hp
import sys
import pandas
import numpy.ma as ma
import scipy.spatial as sp
from astropy import units as u
from astropy.coordinates import SkyCoord

def crossmatch(ddc,dra,rad,ddc_out):
    arr = []
    for i in range(ddc.shape[0]):
        for item in ddc_out[i]:
            dist = np.sqrt((ddc[i][item]*ddc[i][item] + dra[i][item]*dra[i][item]))
            if dist < rad[i]:
                arr = arr + [item]
    return arr

test = pandas.read_csv('full_arr.csv')
print(test)

x = test.lat
y = test.lon

a_tuples = [(test.ra[i], test.dec[i]) for i in range(len(test['dec']))]
a_arrs = (np.array([v[0] for v in a_tuples]), np.array([v[1] for v in a_tuples]))
x, y = a_arrs
points = np.c_[x.ravel(), y.ravel()]

tree = sp.KDTree(points, leafsize=5)

csvFile = pandas.read_csv('lvgs_catalog.csv')
rad = np.float32(csvFile.ORAD*60*2)
dec = np.array(csvFile.DEC)
ra = np.array(csvFile.RA)
c = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
c = c.galactic
lon = np.array(np.float32(c.l.degree))
lat = np.array(np.float32(c.b.degree))
ddc = np.float32(np.abs((np.array(y) - lon[:, np.newaxis])*3600))
dra = np.float32(np.abs((np.array(x) - lat[:, np.newaxis])*3600))
ddc_out = []
for i in range(ddc.shape[0]):
    row = ddc[i,:]
    row_ra = dra[i,:]
    ddc_out.append(np.where((row<rad[i])&(row_ra<rad[i]))[0])
ar2 = crossmatch(ddc,dra,rad,ddc_out)
dec = None
ra = None
rad = None
ddc = None
dra = None
ddc_out = None
print(len(ar2))

csvFile = pandas.read_csv('hii_regions_catalog.csv')
rad = np.array(csvFile.RADIUS*2)
rad_arr = np.array_split(rad, 6)
dec = np.array(csvFile.DEC)
dec_arr = np.array_split(dec, 6)
ra = np.array(csvFile.RA)
ra_arr = np.array_split(ra, 6)
for i in range(len(rad_arr)):
    ra = ra_arr[i]
    dec = dec_arr[i]
    rad = rad_arr[i]
    c = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
    c = c.galactic
    lon = np.array(np.float32(c.l.degree))
    lat = np.array(np.float32(c.b.degree))
    ddc = np.float32(np.abs((np.array(y) - lon[:, np.newaxis])*3600))
    dra = np.float32(np.abs((np.array(x) - lat[:, np.newaxis])*3600))
    ddc_out = []
    for i in range(ddc.shape[0]):
        row = ddc[i,:]
        row_ra = dra[i,:]
        ddc_out.append(np.where((row<rad[i])&(row_ra<rad[i]))[0])
    ar2 = ar2 + crossmatch(ddc,dra,rad,ddc_out)
    print(len(ar2))
    dec = None
    ra = None
    rad = None
    ddc = None
    dra = None
    ddc_out = None


csvFile = pandas.read_csv('pn_catalog.csv')
rad = np.float32(csvFile.RADIUS*2)
dec = np.array(csvFile.DEC)
ra = np.array(csvFile.RA)
c = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
c = c.galactic
lon = np.array(np.float32(c.l.degree))
lat = np.array(np.float32(c.b.degree))
ddc = np.float32(np.abs((np.array(y) - lon[:, np.newaxis])*3600))
dra = np.float32(np.abs((np.array(x) - lat[:, np.newaxis])*3600))
ddc_out = []
for i in range(ddc.shape[0]):
    row = ddc[i,:]
    row_ra = dra[i,:]
    ddc_out.append(np.where((row<rad[i])&(row_ra<rad[i]))[0])
ar2 = ar2 + crossmatch(ddc,dra,rad,ddc_out)
dec = None
ra = None
rad = None
ddc = None
dra = None
ddc_out = None
print(len(ar2))

#This part takes ~3 days to run
#has little impact (cuts ~50,000 AGN)
csvFile = pandas.read_csv('2MASS_XSC.dat', sep='\s+')
rad = csvFile.RAD*2
dec = csvFile.DEC
ra = csvFile.RA
c = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
c = c.galactic
lon = np.array(np.float32(c.l.degree))
lat = np.array(np.float32(c.b.degree))

#Cut out values that are in the galactic plane/center
#Trying to reduce runtime
ind = np.where((lat<10) & (lat>-10))
lon = np.delete(lon,ind)
lat = np.delete(lat,ind)
rad = np.delete(rad,ind)
print(len(lon))
ind = np.where(lon*lon+lat*lat<900)
lon = np.delete(lon,ind)
lat = np.delete(lat,ind)
rad = np.delete(rad,ind)
print(len(lon))


rad_arr = np.array_split(rad, 1600)
lat_arr = np.array_split(lat, 1600)
lon_arr = np.array_split(lon, 1600)
for i in range(len(rad_arr)):
    ra = np.array(ra_arr[i])
    dec = np.array(dec_arr[i])
    rad = np.array(rad_arr[i])
    ddc = np.float32(np.abs((np.array(y) - dec[:, np.newaxis])*3600))
    dra = np.float32(np.abs((np.array(x) - ra[:, np.newaxis])*3600))
    ddc_out = []
    for i in range(ddc.shape[0]):
        row = ddc[i,:]
        row_ra = dra[i,:]
        ddc_out.append(np.where((row<rad[i])&(row_ra<rad[i]))[0])
    ar2 = ar2 + crossmatch(ddc,dra,rad,ddc_out)
    print(len(ar2))
    dec = None
    ra = None
    rad = None
    ddc = None
    dra = None
    ddc_out = None


csvFile = pandas.read_csv('result_bright.csv')
rad = np.float32(np.sqrt(csvFile.Area/np.pi))*3600*2
dec = np.array(csvFile._DE_icrs)
ra = np.array(csvFile._RA_icrs)
c = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
c = c.galactic
lon = np.array(np.float32(c.l.degree))
lat = np.array(np.float32(c.b.degree))
ddc = np.float32(np.abs((np.array(y) - lon[:, np.newaxis])*3600))
dra = np.float32(np.abs((np.array(x) - lat[:, np.newaxis])*3600))
ddc_out = []
for i in range(ddc.shape[0]):
    row = ddc[i,:]
    row_ra = dra[i,:]
    ddc_out.append(np.where((row<rad[i])&(row_ra<rad[i]))[0])
ar2 = ar2 + crossmatch(ddc,dra,rad,ddc_out)
dec = None
ra = None
rad = None
ddc = None
dra = None
ddc_out = None
print(len(ar2))

csvFile = pandas.read_csv('result_dark.csv')
rad = np.float32(np.sqrt(csvFile.Area/np.pi))*3600
dec = np.array(csvFile._DE_icrs)
ra = np.array(csvFile._RA_icrs)
c = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
c = c.galactic
lon = np.array(np.float32(c.l.degree))
lat = np.array(np.float32(c.b.degree))
ddc = np.float32(np.abs((np.array(y) - lon[:, np.newaxis])*3600))
dra = np.float32(np.abs((np.array(x) - lat[:, np.newaxis])*3600))
ddc_out = []
for i in range(ddc.shape[0]):
    row = ddc[i,:]
    row_ra = dra[i,:]
    ddc_out.append(np.where((row<rad[i])&(row_ra<rad[i]))[0])
ar2 = ar2 + crossmatch(ddc,dra,rad,ddc_out)
dec = None
ra = None
rad = None
ddc = None
dra = None
ddc_out = None
print(len(ar2))

print(len(test))
test.drop(labels=ar2, inplace = True)
test = test[['w1','w2','designation','ra','dec','ring256','lat','lon']]
print(len(test))
test.to_csv('full_cut.csv', index = False)