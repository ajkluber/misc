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
     
    pylab.hist2d(Q,qpath,bins=nbins,norm=LogNorm())
    pylab.xlabel("Folding progress Q",fontsize=18)
    pylab.ylabel("Pathway Qpath",fontsize=18)
    #cbar = pylab.colorbar()
    #cbar.set_label("Free energy ($k_BT_f$)",fontsize=18)
    pylab.title("%s it %d  Pathway compared to Vanilla" % (name,iteration),fontsize=18)
    pylab.savefig("qpathvsQ.pdf")
    pylab.savefig("qpathvsQ.png")

    pylab.show()
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

        C[contacts[i,1] - 1,contacts[i,1] - 1] = 

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
