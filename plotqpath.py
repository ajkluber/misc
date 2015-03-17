from matplotlib.colors import LogNorm 
import pylab

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


def plot_Q_vs_Qpath(name,iteration,nbins=50):
    os.chdir("%s/iteration_%d" % (name,iteration))

    temps = [ x.rstrip("\n") for x in open("long_temps_last","r").readlines() ]

    for i in range(len(temps)):
        if i == 0:
            qpath = np.loadtxt("%s/qpath.dat" % temps[i])
            Q = np.loadtxt("%s/Q.dat" % temps[i])
        else:
            qpath = np.concatenate((qpath,np.loadtxt("%s/qpath.dat" % temps[i])))
            Q = np.concatenate((Q,np.loadtxt("%s/Q.dat" % temps[i])))

    early = np.loadtxt("early_conts",dtype=int)
    late = np.loadtxt("late_conts",dtype=int)

    n_contacts = len(early) + len(late)

    maxqpath = max(qpath)
    minqpath = min(qpath)
    maxQ = max(Q)
    minQ = min(Q)
     
    Hxy, xedges, yedges = np.histogram2d(Q,qpath,bins=nbins)
    xcenters = 0.5*(xedges[1:] + xedges[:-1])
    ycenters = 0.5*(yedges[1:] + yedges[:-1])
    X,Y = np.meshgrid(xcenters,ycenters)

    pmf = -np.log(Hxy)
    pmf[pmf != np.NaN] -= pmf[pmf != np.NaN].min()

    maskpmf = np.ma.array(pmf,mask=np.isnan(pmf))

    #plt.contourf(X,Y,maskpmf.T)
    plt.contourf(X,Y,maskpmf.T,levels=np.arange(0,10,1))
    plt.xlabel("Folding progress $Q$",fontsize=18)
    plt.ylabel("Pathway $Q_{path}$",fontsize=18)
    
    #plt.ylim(-maxqpath - 1,maxqpath + 1)
    plt.ylim(-50,60)
    cbar = plt.colorbar()
    cbar.set_label("Free energy ($k_BT_f$)",fontsize=18)
    if iteration == 0:
        plt.title("Homogeneous model pathway",fontsize=18)
    else:
        plt.title("Heterogeneous model pathway",fontsize=18)
    plt.savefig("qpathvsQ.pdf")
    plt.savefig("qpathvsQ.png")
    plt.show()

    os.chdir("../..")

def save_legend(name):
    os.chdir("%s/iteration_0" % name)
    
    early = np.loadtxt("early_conts")
    late = np.loadtxt("late_conts")
    
    T = open("long_temps_last","r").readlines()[0].rstrip("\n")
    contacts = np.loadtxt("%s/native_contacts.ndx" % T, skiprows=1)
    n_conts = len(contacts)

    C = np.zeros((n_conts,n_conts),float)
    for i in range(n_conts):
        C[contacts[i,1] - 1,contacts[i,1] - 1] = 0

    os.chdir("../..")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('--name', type=str, required=True, help='Name of protein to plot.')
    parser.add_argument('--iteration', type=int, required=True, help='Iteration to plot.')
    parser.add_argument('--nbins', type=int, default=50, help='Num of bins.')
    args = parser.parse_args()

    name = args.name
    iteration = args.iteration
    nbins = args.nbins

    #save_legend(name)

    plot_Q_vs_Qpath(name,iteration,nbins=nbins)
