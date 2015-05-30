""" Estimate energetic and entropic fluctuations

Description
-----------
    Cho, Wolynes describes a simple metric for comparing energetic and entropic
fluctuations in structure based models.

"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import model_builder as mdb
import project_tools as pjt
from project_tools.parameter_fitting.util.util import *

global GAS_CONSTANT_KJ_MOL
GAS_CONSTANT_KJ_MOL = 0.0083144621

def energy_entropy_fluctuations(name):

    model, fitopts = mdb.inputs.load_model(name)

    os.chdir("%s/iteration_%d" % (name,fitopts["iteration"]))

    Tlist = [ x.rstrip("\n") for x in open("long_temps_last","r").readlines() ]

    trajfiles = [ "%s/traj.xtc" % x for x in Tlist ]
    native = "%s/Native.pdb" % Tlist[0]
    rij = get_rij(model,trajfiles,native)

    # Get contacts
    Q = np.concatenate([ np.loadtxt("%s/Q.dat" % x) for x in Tlist ])
    n_bins = 30
    n,bins = np.histogram(Q,bins=n_bins)
    Qavg = 0.5*(bins[1:] + bins[:-1])/float(max(Q))

    # Get contact energy
    Eij = get_Vp_for_state(model,rij,np.ones(rij.shape[0],bool),rij.shape[0])

    eps = model.model_param_values[1::2]
    Eij *= eps
    #Eijtotal = np.sum(Eij,axis=1)
    #Eijmean = np.mean(Eij,axis=1)

    # Entropic cost depends on loop length
    lp = (7.*0.38)  # lp = persistence length                                                        
    pair_dists = np.array([ x[0] for x in model.pairwise_other_parameters[1::2] ])   
    deltaV = ((0.5*3.*np.pi)**(3./2.))*((4.*np.pi/3.)*(pair_dists**3))/(lp**3)      
    r0 = np.mean(pair_dists)
    #deltaV = ((0.5*3.*np.pi)**(3./2.))*((4.*np.pi/3.)*(r0**3))/(lp**3)      
    loops = model.pairs[1::2,1] - model.pairs[1::2,0]

    #################################################################
    # Energetic and entropic fluctuations
    #################################################################
    #Sij = np.log(deltaV/(loops**(3./2.)))                             # Jacobson-Stockmayer formula
    #Sij = np.log(deltaV/(Qavg[i]**(3./2.)))                           # Flory mean field formula
    #Sij = np.log(deltaV/((loops**(-3./2.)) + (Qavg[i]**(-3./2.))))    # Shoemaker interpolation

    E = np.zeros(n_bins)
    S = np.zeros(n_bins)
    dE2 = np.zeros(n_bins)
    dS2 = np.zeros(n_bins)
    dEdS = np.zeros(n_bins)
    for i in range(n_bins):
        bin_frames = ((Q > bins[i]).astype(int)*(Q <= bins[i + 1]).astype(int)).astype(bool)
        Econtacts_sum = np.sum(Eij[bin_frames],axis=1)
        E[i] = np.mean(Esum_frames)
        dE2[i] = np.mean(np.std(Eij[bin_frames],axis=1)**2)

        Sij = np.log(deltaV/((loops**(-3./2.)) + (Qavg[i]**(-3./2.))))     # Shoemaker interpolation
        contacts = (rij[bin_frames,1::2] <= 1.2*r0)
        
        Scontacts_sum = np.sum(contacts*Sij,axis=1)
        S[i] = np.mean(Ssum_frames)
        dS2[i] = np.mean(np.std(contacts*Sij,axis=1)**2)

        map(np.dot,(Eij.T - Econtacts_sum),(Sij.T - Scontacts_sum))


    if not os.path.exists("plots"):
        os.mkdir("plots")

    plt.figure()
    plt.plot(Qavg,Efluct,'r',lw=2)
    plt.xlabel("Q")
    plt.ylabel("$\\langle \\delta \\epsilon^2 \\rangle")
    plt.title("Fluctuations in contact energy")
    plt.savefig("plots/Energy_fluct.pdf")
    plt.savefig("plots/Energy_fluct.png")

    plt.figure()
    plt.plot(Qavg,Efluct)
    plt.xlabel("$Q$")
    plt.ylabel("$\\langle \\delta S^2 \\rangle")
    plt.title("Fluctuations in contact entropy")
    plt.savefig("plots/Entropy_fluct.pdf")
    plt.savefig("plots/Entropy_fluct.png")
    plt.show()

    os.chdir("../..")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('--name', type=str, required=True, help='Name of protein to plot.')
    parser.add_argument('--iteration', type=int, default=None, help='Iteration to plot.')
    parser.add_argument('--nbins', type=int, default=30, help='Num of bins.')
    args = parser.parse_args()

    name = args.name
    iteration = args.iteration
    n_bins = args.nbins
    #name = "S6"

    #energy_entropy_fluctuations(name)

    if iteration is None:
        model, fitopts = mdb.inputs.load_model(name)
    else:
        model, fitopts = mdb.inputs.load_model(name + "_" + str(args.iteration))

    pairs = model.pairs[1::2]

    os.chdir("%s/iteration_%d" % (name,fitopts["iteration"]))

    Tlist = [ x.rstrip("\n") for x in open("long_temps_last","r").readlines() ]
    beta = 1./(GAS_CONSTANT_KJ_MOL*float(Tlist[0].split("_")[0]))

    trajfiles = [ "%s/traj.xtc" % x for x in Tlist ]
    native = "%s/Native.pdb" % Tlist[0]
    rij = get_rij(model,trajfiles,native)

    bounds, state_labels = get_state_bounds()
    bounds = [0] + bounds + [model.n_pairs]

    # Get state boundaries
    U,TS,N,Uframes,TSframes,Nframes = concatenate_state_indicators(Tlist,bounds,coord="Q.dat")

    # Get contacts
    Q = np.concatenate([ np.loadtxt("%s/Q.dat" % x) for x in Tlist ])
    #n_bins = 30
    n,bins = np.histogram(Q,bins=n_bins)
    Qavg = 0.5*(bins[1:] + bins[:-1])/float(max(Q))

    # Get contact energy
    eps = model.model_param_values[1::2]
    Eij = beta*eps*get_Vp_for_state(model,rij,np.ones(rij.shape[0],bool),rij.shape[0])
    #Eijtotal = np.sum(Eij,axis=1)
    #Eijmean = np.mean(Eij,axis=1)

    avgEij_U = np.mean(Eij[U,:],axis=0)
    avgEij_TS = np.mean(Eij[TS,:],axis=0)
    avgEij_N = np.mean(Eij[N,:],axis=0)

    stdEij_U = np.std(Eij[U,:],axis=0)
    stdEij_TS = np.std(Eij[TS,:],axis=0)
    stdEij_N = np.std(Eij[N,:],axis=0)

    # Entropic cost depends on loop length
    lp = (7.*0.38)  # lp = persistence length is about 7 residues
    pair_dists = np.array([ x[0] for x in model.pairwise_other_parameters[1::2] ])   
    deltaV = ((0.5*3.*np.pi)**(3./2.))*((4.*np.pi/3.)*(pair_dists**3))/(lp**3)      
    r0 = np.mean(pair_dists)
    #deltaV = ((0.5*3.*np.pi)**(3./2.))*((4.*np.pi/3.)*(r0**3))/(lp**3)      
    loops = model.pairs[1::2,1] - model.pairs[1::2,0]

    dcontact_E = np.zeros((n_bins,Eij.shape[1]))
    dcontact_S = np.zeros((n_bins,Eij.shape[1]))
    dQ2 = np.zeros((n_bins,Eij.shape[1]))
    E = np.zeros(n_bins)
    S = np.zeros(n_bins)
    dE2 = np.zeros(n_bins)
    dS2 = np.zeros(n_bins)
    dEdS = np.zeros(n_bins)
    for i in range(n_bins):
    #for i in [0]:
        bin_frames = ((Q > bins[i]).astype(int)*(Q <= bins[i + 1]).astype(int)).astype(bool)
        dcontact_E[i,:] = np.std(Eij[bin_frames,:],axis=0)**2

        Econtacts_sum = np.sum(Eij[bin_frames,:],axis=1)
        E[i] = np.mean(Econtacts_sum)
        dE2[i] = np.mean(np.std(Eij[bin_frames,:],axis=1)**2)

        Sij = np.log(deltaV/((loops**(-3./2.)) + (Qavg[i]**(-3./2.))))     # Shoemaker interpolation
        contacts = (rij[bin_frames,1::2] <= 1.2*pair_dists)

        dQ2[i,:] = np.std(contacts,axis=0)**2

        Scontacts = contacts*Sij
        Scontacts_sum = np.sum(Scontacts,axis=1)
        S[i] = np.mean(Scontacts_sum)
        dS2[i] = np.mean(np.std(Scontacts,axis=1)**2)
        dcontact_S[i,:] = np.std(Scontacts,axis=0)**2

        dE = (Eij[bin_frames,:].T - np.mean(Eij[bin_frames,:],axis=1)).T
        dS = (Scontacts.T - np.mean(Scontacts,axis=1)).T
        dEdS[i] = np.mean(map(np.dot,dE,dS))


    baseline_fluct = dcontact_E[0,:] + dcontact_E[-1,:]


    relative_TS_fluct = (stdEij_TS**2)/(0.5*(stdEij_U**2 + stdEij_N**2))

    if not os.path.exists("fluct"):
        os.mkdir("fluct")
    os.chdir("fluct")


    np.savetxt("Qavg.dat",Qavg)
    np.savetxt("dE2.dat",dE2)
    np.savetxt("dS2.dat",dS2)
    np.savetxt("dQ2.dat",dQ2)
    np.savetxt("dcontact_E.dat",dcontact_E)
    np.savetxt("dcontact_S.dat",dcontact_E)

    Y = dcontact_E/(0.5*(stdEij_U**2 + stdEij_N**2))

    plt.figure()
    for i in range(Eij.shape[1]):
        plt.plot(Qavg,Y[:,i],color=cm.gnuplot2((relative_TS_fluct[i] - relative_TS_fluct.min())/relative_TS_fluct.max()))
    plt.xlabel("$Q$",fontsize=18)
    plt.ylabel("$\\frac{2\\langle\\delta\\epsilon_{ij}^2\\rangle(Q)}{\\langle\\delta\\epsilon_{ij}^2\\rangle_U + \\langle\\delta\\epsilon_{ij}^2\\rangle_N}$",fontsize=22)
    #plt.title("$\\frac{2\\langle\\delta\\epsilon_{ij}^2\\rangle}{\\left(\\langle\\delta\\epsilon_{ij}^2\\rangle_U + \\langle\\delta\\epsilon_{ij}^2\\rangle_N\\right)}$")
    plt.title("Energy Fluctuations Relative to average of U and N")
    plt.savefig("eps_fluct_relative_U_and_N.png")
    plt.savefig("eps_fluct_relative_U_and_N.pdf")
    plt.savefig("eps_fluct_relative_U_and_N.eps")


    Z = dcontact_E/(stdEij_N**2)
    plt.figure()
    for i in range(Eij.shape[1]):
        plt.plot(Qavg,Z[:,i],color=cm.gnuplot2((relative_TS_fluct[i] - relative_TS_fluct.min())/relative_TS_fluct.max()))
    plt.xlabel("Q")
    #plt.ylabel("$\\langle\\delta\\epsilon_{ij}^2\\rangle / \\langle\\delta\\epsilon_{ij}^2\\rangle_N $")
    plt.title("Energy Fluctuations Relative to N")
    #plt.title("$\\langle\\delta\\epsilon_{ij}^2\\rangle / \\langle\\delta\\epsilon_{ij}^2\\rangle_N $")
    plt.ylabel("$\\frac{\\langle\\delta\\epsilon_{ij}^2\\rangle}{\\langle\\delta\\epsilon_{ij}^2\\rangle_N}$",fontsize=18)
    plt.savefig("TS_eps_fluct_relative_N.png")
    plt.savefig("TS_eps_fluct_relative_N.pdf")
    plt.savefig("TS_eps_fluct_relative_N.eps")

    #cmap = plt.get_cmap("spectral")
    #cmap = plt.get_cmap("bone_r")
    cmap = plt.get_cmap("gnuplot2")
    #cmap.set_bad(color='w',alpha=1.)
    cmap.set_bad(color='gray',alpha=1)

    plt.figure()
    C = np.zeros((model.n_residues,model.n_residues),float)*np.nan
    for i in range(Eij.shape[1]):
        C[ pairs[i,1] - 1,pairs[i,0] - 1] = (relative_TS_fluct[i] - relative_TS_fluct.min())/relative_TS_fluct.max()
    C = np.ma.masked_invalid(C)
    plt.pcolormesh(C,cmap=cmap,vmin=0,vmax=1,edgecolors='None')
    plt.xlim(0,model.n_residues)
    plt.ylim(0,model.n_residues)
    plt.xticks(range(0,model.n_residues,10))
    plt.yticks(range(0,model.n_residues,10))
    cbar = plt.colorbar()
    cbar.set_label("TS Fluctuations (% max)")
    plt.xlabel("Residue i")
    plt.ylabel("Residue j")
    plt.title("TS energy fluctuations")
    plt.savefig("TS_eps_fluct_relative.png")
    plt.savefig("TS_eps_fluct_relative.pdf")
    plt.savefig("TS_eps_fluct_relative.eps")

    plt.show()


    os.chdir("../../..")
