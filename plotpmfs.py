import argparse
import numpy as np
import matplotlib.pyplot as plt

from bootFQ import get_F_with_error

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('--data', type=str, default="Q.dat", help='Name of timeseries to analyze.')
    parser.add_argument('--n_bins', type=int, default=25, help='Optional. number of bins along coordinate.')
    parser.add_argument('--n_histos', type=int, default=100, help='Optional. number of bootstrapping histograms.')
    parser.add_argument('--stride', type=int, default=10, help='Optional. number of frames to skip for subsampling.')
    parser.add_argument('--title', type=str, default="", help='Optional. Title for plot.')
    parser.add_argument('--saveas', type=str, default="None", help='Optional. Filename to save plot.')
    args = parser.parse_args()

    data = args.data 
    n_bins = args.n_bins 
    n_histos = args.n_histos
    stride = args.stride
    title = args.title
    saveas = args.saveas

    coord = data.split(".")[0]

    temps = [ x.rstrip("\n") for x in open("long_temps_last","r").readlines() ]
    colors = ['b','r','g','k','cyan','magenta']
        
    for i in range(len(temps)):
        filename = "%s/%s" % (temps[i],data)
        F,F_err,bin_centers = get_F_with_error(filename,n_bins,10,1)
        plt.plot(bin_centers,F,lw=2,color=colors[i],label=temps[i])
        plt.fill_between(bin_centers,F + F_err,F - F_err,facecolor=colors[i],alpha=0.25)

    plt.legend()
    plt.xlabel("%s" % coord,fontsize=20)
    plt.ylabel("F(%s) (k$_B$)" % coord,fontsize=20)
    plt.title(title)
    if saveas != "None":
        plt.savefig(saveas)
    plt.show()
