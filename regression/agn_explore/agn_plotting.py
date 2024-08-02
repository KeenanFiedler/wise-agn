import numpy as np
import pandas as pd
import bisect

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.mlab as mlab

import shapely
from shapely.geometry import Point, LineString
from shapely.geometry.polygon import Polygon

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

    """
    ind = np.where((desi.zerr<0.005) & (desi.zerr>0) & (desi.z>0.1) & (desi.z<3) & (desi.spectype =='QSO'))[0]
    desi = desi.loc[ind]
    desi.to_csv('desi.csv', index = None)
    ind = np.where((sdss.zerr<0.005) & (sdss.zerr>0) & (sdss.z>0.1) & (sdss.z<3) & (sdss.sourcetype =='QSO'))[0]
    sdss = sdss.loc[ind]
    sdss.to_csv('sdss.csv', index = None)
    """
    #z = np.array(sdss.z)
    #z = np.append(z, desi.z)
    width=0.05
    w21_min, w21_max = 0.5,3
    w32_min, w32_max = 1.25,5.5

    range_ = [[w32_min,w32_max],[w21_min,w21_max]]

    cx = 0
    cy = 0
    grid_cells = []
    for y0 in np.arange(w21_min, w21_max+width, width):
        cy += 1
        for x0 in np.arange(w32_min, w32_max+width, width):
            x1 = x0+width
            y1 = y0+width
            new_cell = shapely.geometry.box(x0, y0, x1, y1)
            grid_cells.append(new_cell)
            if y0 == 0.5:
                cx += 1
    w1 = np.array(sdss.w1)
    w1 = np.append(w1, desi.w1)
    w2 = np.array(sdss.w2)
    w2 = np.append(w2, desi.w2)
    w3 = np.array(sdss.w3)
    w3 = np.append(w3, desi.w3)
    xarr = w2-w3
    yarr = w1-w2
    H,_,_ = np.histogram2d(xarr,yarr,bins=[cx,cy],range=range_)

    cmap = plt.cm.jet
    norm = mpl.colors.LogNorm()
    clabel = r'counts / mag$^2$'
    intp = 'none'
    def make_panel(ax,data,extent=None,title='',ylabel='',xlabel=''):
        im = ax.imshow(data,origin='lower',extent=extent,cmap=cmap,interpolation=intp,norm=norm)
        cb = plt.colorbar(im)
        cb.set_label(clabel)
        plt.title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    extent = [1.25,5.5,0.5,3]
    cellarea = width*width
    fig, ax = plt.subplots(2,1)
    fig.set_figwidth(4.5)
    fig.suptitle('Data vs CLUMPY models')
    ax1 = ax[0]
    ax2 = ax[1]
    make_panel(ax1,H.T/cellarea,extent=extent,title='Data',ylabel='W1-W2')
    colors = np.load('clumpy_colors.npy')
    w21 = colors[:,0]
    w23 = colors[:,1]
    H,_,_ = np.histogram2d(w23,w21,bins=[cx,cy],range=range_)
    make_panel(ax2,H.T/cellarea,extent=extent,title='CLUMPY',ylabel='W1-W2',xlabel='W2-W3')
    plt.tight_layout()
    plt.savefig('CLUMPYvsData.png', dpi=1000)

    """
    make_redshift_plot(desi, 'desi')
    make_histogram(w1, z)
    draw_normal(z,1)
    """

main()
