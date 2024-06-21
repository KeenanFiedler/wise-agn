import matplotlib.pyplot as plt
import matplotlib

from dl import authClient as ac, queryClient as qc, storeClient as sc

import numpy as np
import healpy as hp
import sys
import pandas
import numpy.ma as ma
import scipy.spatial as sp

query = """
    SELECT *
    FROM mydb://allwise_center_errorcut
    """
test = qc.query(sql=query, fmt = 'pandas',timeout=1000000000)
print(test)

a_tuples = [(test.ra[i], test.dec[i]) for i in range(len(test['dec']))]
a_arrs = (np.array([v[0] for v in a_tuples]), np.array([v[1] for v in a_tuples]))
x, y = a_arrs
points = np.c_[x.ravel(), y.ravel()]

tree = sp.KDTree(points, leafsize=5)


csvFile = pandas.read_csv('lvgs_catalog.csv')
rad = csvFile.ORAD/60
a = rad*rad*np.pi
print('Area for LVG: ' + str(sum(a)))
dec = csvFile.DEC
ra = csvFile.RA
centers = np.c_[ra, dec]
contained = tree.query_ball_point(centers, rad)
arr = []
for l in contained:
    arr = arr + l
print('Total number of removed gals: ' + str(len(arr)))


csvFile = pandas.read_csv('hii_regions_catalog.csv')
rad = csvFile.RADIUS/1800
a = rad*rad*np.pi
print('Area for HII: ' + str(sum(a)))
dec = csvFile.DEC
ra = csvFile.RA
centers = np.c_[ra, dec]
contained = tree.query_ball_point(centers, rad)
for l in contained:
    arr = arr + l
print('Total number of removed gals: ' + str(len(arr)))


csvFile = pandas.read_csv('pn_catalog.csv')
rad = csvFile.RAD/1800
a = rad*rad*np.pi
print('Area for PN: ' + str(sum(a)))
dec = csvFile.DEC
ra = csvFile.RA
centers = np.c_[ra, dec]
contained = tree.query_ball_point(centers, rad)
for l in contained:
    arr = arr + l
print('Total number of removed gals: ' + str(len(arr)))


csvFile = pandas.read_csv('2MASS_XSC.dat', sep='\s+')
rad = csvFile.RAD/1800
a = rad*rad*np.pi
print('Area for 2MASS: ' + str(sum(a)))
dec = csvFile.DEC
ra = csvFile.RA
centers = np.c_[ra, dec]
contained = tree.query_ball_point(centers, rad)
for l in contained:
    arr = arr + l
print('Total number of removed gals: ' + str(len(arr)))


csvFile = pandas.read_csv('result_bright.csv')
rad = np.sqrt(csvFile.Area/np.pi)
print('Area for Bright Nebula: ' + str(sum(csvFile.Area)))
dec = csvFile._DE_icrs
ra = csvFile._RA_icrs
centers = np.c_[ra, dec]
contained = tree.query_ball_point(centers, rad)
for l in contained:
    arr = arr + l
print('Total number of removed gals: ' + str(len(arr)))


csvFile = pandas.read_csv('result.csv')
rad = np.sqrt(csvFile.Area/np.pi)
print('Area for Dark Nebula: ' + str(sum(csvFile.Area)))
dec = csvFile._DE_icrs
ra = csvFile._RA_icrs
centers = np.c_[ra.ravel(), dec.ravel()]
contained = tree.query_ball_point(centers, rad)
for l in contained:
    arr = arr + l
print('Total number of removed gals: ' + str(len(arr)))

print(len(test))
test.drop(labels=arr, inplace = True)
print(len(test))