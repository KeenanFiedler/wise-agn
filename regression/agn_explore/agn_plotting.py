import numpy as np
import pandas as pd
import bisect

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.mlab as mlab

from scipy.stats import norm
from scipy.stats import skewnorm
from scipy.stats import ecdf

def make_redshift_plot(data, survey):
    z = np.array(data.z)
    zerr = np.array(data.zerr)
    ind = np.where((zerr<0.005) & (zerr>0) & (z>0.1))[0]
    z=z[ind]
    zerr=zerr[ind]

    plt.scatter(z, zerr/z, s=0.5, linewidths=0)
    plt.xlabel('Redshift')
    plt.ylabel('Redshift Error/Redshift')
    plt.title('Redshift vs Error')
    plt.savefig(survey + '_redshift.png', dpi=300)
    plt.clf()

def make_histogram(w1, z):
    (w1_mu, w1_sig,w1_skew) = skewnorm.fit(w1)
    (z_mu, z_sig) = norm.fit(z)

    _, bins, _ = plt.hist(w1, bins=40, density=True)

    # Over plot line of skewnorm pdf
    w1_norm = skewnorm.pdf(bins, w1_mu, w1_sig, w1_skew)
    plt.plot(bins, w1_norm, 'r--')

    # Over plot histogram of random samples
    random_draws = skewnorm.rvs(w1_mu, w1_sig, w1_skew, size=100000)
    plt.hist(random_draws, bins=40, alpha=0.5, density = True)

    plt.xlabel('W1 Magnitude')
    plt.title('W1 Magnitude Histogram')
    plt.savefig('w1_hist.png', dpi=300)
    plt.clf()

    _, bins, _ = plt.hist(z, bins=40, density=True)

    # Get ECDF
    z_ecdf = ecdf(z)

    # Create cdf values for each redshift value
    x = np.linspace(0,3,1000)
    y = z_ecdf.cdf.evaluate(x)

    # Plot PDF on top
    pdf_y =z_ecdf.cdf.evaluate(bins)
    pdf = np.gradient(pdf_y,bins)
    plt.plot(bins, pdf, 'r--')

    # Draw randomly from cdf to create pdf
    x_est = []
    for i in range(10**6):
        rand = np.random.uniform(0,1)
        index = bisect.bisect_right(y, rand)
        x_est.append(x[index])
    plt.hist(x_est, bins=40, density = True, alpha=0.5)

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

    z = np.array(sdss.z)
    z = np.append(z, desi.z)
    w1 = np.array(sdss.t1_w1)
    w1 = np.append(w1, desi.t1_w1)


    make_redshift_plot(desi, 'desi')
    make_histogram(w1, z)
    draw_normal(z,1)

main()
