""" Determine errors of PMF along coordinate by bootsrapping. 

Description:
    We can estimate the errors from insufficient sampling of a one-dimensional 
free energy profile by performing bootstrapping. Since barrier crossing is a
rare event we are most interested in estimating the error associated with the
height of the barrier. 
    The bootstrapping procedure uses the observed histogram to generate many
hypothetically observed histograms. The estimated error of the observed 
histogram counts is determined by calculating the variance of the generated
histograms.

Arguments:
  --data	Name of datafile with coordinate timeseries.
  --n_bins	Optional. Number of histogram bins to use along coordinate.
  --n_histos	Optional. Number of bootstrapping histograms to use.

See Also: 
(1) 'Bootstrapping'. Wikipedia. http://en.wikipedia.org/wiki/Bootstrapping_%28statistics%29
"""

import argparse
import time
import numpy as np
import matplotlib.pyplot as plt

def generate_histograms(n_histos,n_bins,n_samples,cumul):
    """ Generate hypothetical histograms with given statistics."""
    boot_histos = np.zeros((n_histos,n_bins),float)
    print "  Generating %d bootstrap histograms:" % n_histos
    for i in range(n_histos):
        if (i % 10) == 0:
            print "    %d" % i 
        for j in range(n_samples):
            p = np.random.rand()
            for n in range(n_bins):
                if (cumul[n] < p) and (p <= cumul[n+1]):
                    boot_histos[i,n] = boot_histos[i,n] + 1.
                    break
    return boot_histos

def get_F_with_error(filename,n_bins,n_histos,stride):

    # Subsample the data.
    q = np.loadtxt(filename)
    q_sub = q[::stride]
    n_samples = len(q_sub)

    # Calculate cumulative probability distribution of coordinate.
    hist,bins = np.histogram(q_sub, bins=n_bins, density=False)
    hist_norm = hist / float(sum(hist))
    cumul = np.asarray([float(sum(hist_norm[:i])) for i in range(len(hist_norm))] + [1.]) 
    F = -np.log(hist_norm)
    F -= min(F) 

    # Estimate errorbars on free energy profile using bootstrapping.
    boot_histos = generate_histograms(n_histos,n_bins,n_samples,cumul)
    boot_pmf = -np.log(boot_histos)
    F_err = np.zeros(n_bins,float)
    for i in range(n_bins):
        F_err[i] = np.std(boot_pmf[:,i])
    bin_centers = 0.5*(bins[:-1]+bins[1:])

    return F,F_err,bin_centers

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('--data', type=str, default="Q.dat", help='Name of timeseries to analyze.')
    parser.add_argument('--n_bins', type=int, default=25, help='Optional. number of bins along coordinate.')
    parser.add_argument('--n_histos', type=int, default=100, help='Optional. number of bootstrapping histograms.')
    parser.add_argument('--stride', type=int, default=10, help='Optional. number of frames to skip for subsampling.')
    parser.add_argument('--saveas', type=str, default="None", help='Optional. Filename to save plot.')
    args = parser.parse_args()

    filename = args.data
    n_bins = args.n_bins 
    n_histos = args.n_histos
    stride = args.stride
    saveas = args.saveas
    
    F,F_err,bin_centers = get_F_with_error(filename,n_bins,n_histos,stride)

    # Plot free energy profile with errorbars
    coord = filename.split(".")[0]
    #plt.errorbar(bin_centers,F,yerr=F_err,lw=1.5,color='b',ecolor='b',elinewidth=1.5)
    plt.plot(bin_centers,F,lw=2,color='b')
    plt.fill_between(bin_centers,F + F_err,F - F_err,facecolor='b',alpha=0.3)
    plt.title("%d bins, %d histograms, %d stride" % (n_bins,n_histos,stride),fontsize=16)
    plt.xlabel("%s" % coord,fontsize=20)
    plt.ylabel("F(%s) (k$_B$T)" % coord,fontsize=20)
    if saveas != "None":
        plt.savefig(saveas)
    plt.show()
