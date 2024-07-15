import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.mlab as mlab

from scipy.stats import norm

def make_redshift_plot(data, survey):
    z = np.array(data.z)
    zerr = np.array(data.zerr)
    ind = np.where((zerr<0.005) & (zerr>0))[0]
    z=z[ind]
    zerr=zerr[ind]

    plt.scatter(z, zerr, s=0.5, linewidths=0)
    plt.xlabel('Redshift')
    plt.ylabel('Redshift Error')
    plt.title('Redshift vs Error')
    plt.savefig(survey + '_redshift.png', dpi=300)
    plt.clf()

def make_histogram(w1, z):
    (w1_mu, w1_sig) = norm.fit(w1)
    (z_mu, z_sig) = norm.fit(z)

    _, bins, _ = plt.hist(w1, bins=40, density=True)

    w1_norm = norm.pdf(bins, w1_mu, w1_sig)
    plt.plot(bins, w1_norm, 'r--', linewidth=2)

    plt.xlabel('W1 Magnitude')
    plt.title('W1 Magnitude Histogram')
    plt.savefig('w1_hist.png', dpi=300)
    plt.clf()

    _, bins, _ = plt.hist(z, bins=40, density=True)

    z_norm = norm.pdf(bins, z_mu, z_sig)
    plt.plot(bins, z_norm, 'r--', linewidth=2)

    plt.xlabel('Redshift')
    plt.title('Redshift Histogram')
    plt.savefig('z_hist.png', dpi=300)
    plt.clf()

def draw_normal(data, amount):
    (mu, sig) = norm.fit(data)
    print(mu)
    print(sig)

    return np.random.normal(mu,sig,amount)

def main():
    desi = pd.read_csv('desi.csv')
    sdss = pd.read_csv('sdss.csv')

    ind_sdss = sdss.z<1
    ind_desi = desi.z<1
    sdss = sdss[ind_sdss]
    desi = desi[ind_desi]
    ind_sdss = sdss.zerr<0.005
    ind_desi = desi.zerr<0.005
    sdss = sdss[ind_sdss]
    desi = desi[ind_desi]
    z = np.array(sdss.z)
    z = np.append(z, desi.z)
    w1 = np.array(sdss.t1_w1)
    w1 = np.append(w1, desi.t1_w1)


    #make_redshift_plot(sdss, 'sdss')
    make_histogram(w1, z)
    draw_normal(z,1)

main()
