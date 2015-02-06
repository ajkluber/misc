import argparse
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('--data', type=str, default="Q.dat", help='Name of timeseries to analyze.')
    parser.add_argument('--n_bins', type=int, default=20, help='Optional. number of bins for histogram.')
    parser.add_argument('--title', type=str, default="", help='Optional. Title for plot.')
    parser.add_argument('--saveas', type=str, default="None", help='Optional. Filename to save plot.')
    args = parser.parse_args()

    filename = args.data 
    coord = filename.split(".")[0]

    Q = np.loadtxt(filename)
    n,bins = np.histogram(Q,bins=args.n_bins)
    F = -np.log(n)
    F -= min(F)
    x = 0.5*(bins[1:] + bins[:-1])
    plt.plot(x,F,'b',lw=2)
    plt.xlabel("%s" % coord,fontsize=20)
    plt.ylabel("F(%s) (k$_B$T)" % coord,fontsize=20)
    plt.title(args.title)
    if args.saveas != "None":
        plt.savefig(args.saveas)
    plt.show()
