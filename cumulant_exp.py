
import os
import argparse
import numpy as np
import matplotlib 
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.special import gamma
from scipy.stats import kstat

import model_builder as mdb
import project_tools as pjt
from project_tools.parameter_fitting.util.util import *

global GAS_CONSTANT_KJ_MOL
GAS_CONSTANT_KJ_MOL = 0.0083144621

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
    states = [U,TS,N]
    state_frames = [Uframes,TSframes,Nframes]

    # Get contacts
    Q = np.concatenate([ np.loadtxt("%s/Q.dat" % x) for x in Tlist ])
    #n_bins = 30
    n,bins = np.histogram(Q,bins=n_bins)
    Qavg = 0.5*(bins[1:] + bins[:-1])/float(max(Q))

    loops = model.pairs[1::2,1] - model.pairs[1::2,0]

    # Get contact energy
    eps = model.model_param_values[1::2]
    Eij = beta*eps*get_Vp_for_state(model,rij,np.ones(rij.shape[0],bool),rij.shape[0])

    avgEij_U = np.mean(Eij[U,:],axis=0)
    avgEij_TS = np.mean(Eij[TS,:],axis=0)
    avgEij_N = np.mean(Eij[N,:],axis=0)

    stdEij_U = np.std(Eij[U,:],axis=0)
    stdEij_TS = np.std(Eij[TS,:],axis=0)
    stdEij_N = np.std(Eij[N,:],axis=0)

    alpha = 1.

    dG_FEP = np.zeros((n_bins,Eij.shape[1]))
    dG_cumulant_exp = np.array([ np.zeros((n_bins,Eij.shape[1])) for x in range(4) ])
    for i in range(n_bins):
    #for i in range(3):
        #bin_frames = states[i]
        bin_frames = ((Q > bins[i]).astype(int)*(Q <= bins[i + 1]).astype(int)).astype(bool)
        n_frames = float(sum(bin_frames))

        dG_FEP[i,:]  = -np.log(np.sum(np.exp(alpha*Eij[bin_frames,:]),axis=0)/n_frames)
        
        print "Bin %4d" % i
        for k in range(1,5):
            for j in range(Eij.shape[1]):
                dG_cumulant_exp[k - 1,i,j] = ((alpha**k)/gamma(k + 1))*kstat(Eij[bin_frames,j],n=k)


    if not os.path.exists("FEPestimate"):
        os.mkdir("FEPestimate")
    os.chdir("FEPestimate")


    for i in range(4):
        np.save("cumulant_exp_term_%d_vs_Q.npy" % i,dG_cumulant_exp[i,:,:])

    raise SystemExit

    dG_fluct_sum = -np.sum(dG_cumulant_exp[1:,:,:],axis=0)
    dG_cumu_mean = -dG_cumulant_exp[0,:,:]

    dG_cumulant_sum = -np.sum(dG_cumulant_exp,axis=0)

    dG_cumulant = np.array([ np.zeros((3,Eij.shape[1])) for x in range(4) ])
    for i in range(3):
        bin_frames = states[i]
        n_frames = float(sum(bin_frames))
        for k in range(1,5):
            for j in range(Eij.shape[1]):
                dG_cumulant[k - 1,i,j] = ((alpha**k)/gamma(k + 1))*kstat(Eij[bin_frames,j],n=k)

    dG_cumulant_U  = -np.sum(dG_cumulant[1:,0,:],axis=0) 
    dG_cumulant_TS = -np.sum(dG_cumulant[1:,1,:],axis=0) 
    dG_cumulant_N  = -np.sum(dG_cumulant[1:,2,:],axis=0) 

    dG_cumulant_U_avg  = -dG_cumulant[0,0,:]
    dG_cumulant_TS_avg = -dG_cumulant[0,1,:]
    dG_cumulant_N_avg  = -dG_cumulant[0,2,:]
    

    #plt.figure()
    #plt.plot(dG_FEP[0,:],dG_cumulant_sum[0,:],'r.')
    #plt.plot(dG_FEP[1,:],dG_cumulant_sum[1,:],'g.')
    #plt.plot(dG_FEP[2,:],dG_cumulant_sum[2,:],'b.')

    #plt.plot([0],[0],'r.',label="U")
    #plt.plot([0],[0],'g.',label="TS")
    #plt.plot([0],[0],'b.',label="N")

    #plt.figure()
    #plt.plot(dG_FEP[0,:],-dG_cumulant[0,0,:]-dG_cumulant[1,0,:],'r.')
    #plt.plot(dG_FEP[1,:],-dG_cumulant[0,1,:]-dG_cumulant[1,1,:],'g.')
    #plt.plot(dG_FEP[2,:],-dG_cumulant[0,2,:]-dG_cumulant[1,2,:],'b.')

    #plt.plot([0],[0],'r.',label="U")
    #plt.plot([0],[0],'g.',label="TS")
    #plt.plot([0],[0],'b.',label="N")

    plt.show()

    raise SystemExit

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

    plt.figure()
    for i in range(Eij.shape[1]):
        plt.plot(Qavg,0.5*(dcontact_E[:,i] - (stdEij_U[i]**2)) - (contact_avgE[:,i] - avgEij_U[i]),color=cm.gnuplot2((relative_TS_fluct[i] - relative_TS_fluct.min())/relative_TS_fluct.max()))
    plt.xlabel("$Q$",fontsize=20)
    plt.ylabel("$\\Delta\\Delta G = \\frac{\\Delta\\sigma^2_{E_i}}{2} - \\Delta\\overline{E}_i$",fontsize=16)
    plt.title("$\\Delta\\Delta G$ model with fluctuations",fontsize=18)
    plt.savefig("ddG_vs_Q_with_fluctuations.png")
    plt.savefig("ddG_vs_Q_with_fluctuations.pdf")
    plt.savefig("ddG_vs_Q_with_fluctuations.eps")

    plt.figure()
    for i in range(Eij.shape[1]):
        plt.plot(Qavg,-(contact_avgE[:,i] - avgEij_U[i]),color=cm.gnuplot2((relative_TS_fluct[i] - relative_TS_fluct.min())/relative_TS_fluct.max()))
    plt.xlabel("$Q$",fontsize=20)
    plt.ylabel("$\\Delta\\Delta G = -\\Delta\\overline{E}_i$",fontsize=16)
    plt.title("$\\Delta\\Delta G$ model without fluctuations",fontsize=18)
    plt.savefig("ddG_vs_Q_without_fluctuations.png")
    plt.savefig("ddG_vs_Q_without_fluctuations.pdf")
    plt.savefig("ddG_vs_Q_without_fluctuations.eps")

    plt.figure()
    for i in range(Eij.shape[1]):
        plt.plot(Qavg,0.5*(dcontact_E[:,i] - (stdEij_U[i]**2)),color=cm.gnuplot2((relative_TS_fluct[i] - relative_TS_fluct.min())/relative_TS_fluct.max()))
    plt.xlabel("$Q$",fontsize=20)
    plt.ylabel("$\\Delta\\Delta G = \\frac{\\Delta\\sigma^2_{E_i}}{2}$",fontsize=16)
    plt.title("$\\Delta\\Delta G$ model only fluctuations",fontsize=18)
    plt.savefig("ddG_vs_Q_only_fluctuations.png")
    plt.savefig("ddG_vs_Q_only_fluctuations.pdf")
    plt.savefig("ddG_vs_Q_only_fluctuations.eps")

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
    
    #plt.show()

    os.chdir("../../..")
