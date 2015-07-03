import argparse
import numpy as np
import matplotlib.pyplot as plt

#from bootFQ import get_F_with_error

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('--data', type=str, default="Q.dat", help='Name of timeseries to analyze.')
    parser.add_argument('--file', type=str, default="long_temps_last", help='File holding temps to plot.')
    parser.add_argument('--n_bins', type=int, default=25, help='Optional. number of bins along coordinate.')
    parser.add_argument('--n_histos', type=int, default=2, help='Optional. number of bootstrapping histograms.')
    parser.add_argument('--stride', type=int, default=1, help='Optional. number of frames to skip for subsampling.')
    parser.add_argument('--title', type=str, default="", help='Optional. Title for plot.')
    parser.add_argument('--all', action='store_true', help='Optional. Concatenate .')
    parser.add_argument('--saveas', type=str, default=None, help='Optional. Filename to save plot.')
    args = parser.parse_args()

    data = args.data 
    file = args.file
    n_bins = args.n_bins 
    n_histos = args.n_histos
    stride = args.stride
    title = args.title
    saveas = args.saveas

    coord = data.split(".")[0]

    temps = [ x.rstrip("\n") for x in open(file,"r").readlines() ]
    colors = ['b','r','g','k','cyan','magenta','salmon','darkgreen']
        
    if args.all:
        # Sort temperatures
        uniq_Tlist = []
        Qlist = []
        print " Gathering unique Tlist"
        for i in range(len(temps)):
            T = temps[i].split("_")[0]
            if T not in uniq_Tlist:
                print " Adding ",T," to ", uniq_Tlist
                uniq_Tlist.append(T)
                Qtemp = np.loadtxt("%s/%s" % (temps[i],data))
                if data == "Q.dat":
                    Qtemp += np.random.normal(size=len(Qtemp))
                Qlist.append(Qtemp)
            else:
                print " Appending ",T," to ", uniq_Tlist
                idx = uniq_Tlist.index(T)
                Qtemp = np.loadtxt("%s/%s" % (temps[i],data))
                if data == "Q.dat":
                    Qtemp += np.random.normal(size=len(Qtemp))
                Qlist[idx] = np.concatenate((Qlist[idx],Qtemp))


        for i in range(len(Qlist)):
            # Subsample the data.
            q_sub = Qlist[i][::stride]
            q_sub += np.random.normal(size=len(q_sub))
            print len(q_sub)
            n,bins = np.histogram(q_sub, bins=n_bins, density=False)
            bin_centers = 0.5*(bins[1:] + bins[:-1])
            F = -np.log(n)
            F -= min(F) 
            plt.plot(bin_centers,F,lw=2,color=colors[i],label=uniq_Tlist[i])
    else:
        for i in range(len(temps)):
            # Subsample the data.
            q = np.loadtxt("%s/%s" % (temps[i],data))
            if data == "Q.dat":
                q += np.random.normal(size=len(q))
            q_sub = q[::stride]

            # Calculate cumulative probability distribution of coordinate.
            n,bins = np.histogram(q_sub, bins=n_bins, density=False)
            bin_centers = 0.5*(bins[1:] + bins[:-1])
            F = -np.log(n)
            F -= min(F) 

            plt.plot(bin_centers,F,lw=2,color=colors[i],label=temps[i])

    plt.legend()
    plt.xlabel("%s" % coord,fontsize=20)
    plt.ylabel("F(%s) (k$_B$T)" % coord,fontsize=20)
    plt.title(title)
    if saveas is not None:
        plt.savefig(saveas)
    plt.show()
