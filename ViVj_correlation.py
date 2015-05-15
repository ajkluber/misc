
import os
import sys
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt

import model_builder as mdb

from project_tools.parameter_fitting.util.util import *

global GAS_CONSTANT_KJ_MOL
GAS_CONSTANT_KJ_MOL = 0.0083144621

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('--name', type=str, required=True, help='Name of directory.')
    parser.add_argument('--iteration', type=int, default=None, help='Optional. Iteration')
    args = parser.parse_args()

    name = args.name
    if args.iteration:
        model,fitopts = mdb.inputs.load_model(name+"_%d" % args.iteration)
    else:
        model,fitopts = mdb.inputs.load_model(name)

    name = model.name
    iteration = fitopts['iteration']

    # Get mutations and fraction of native pairs deleted for each mutation.
    os.chdir("%s/iteration_%d" % (name,iteration))

    Tlist = [ x.rstrip("\n") for x in open("long_temps_last","r").readlines() ]
    beta = 1./(GAS_CONSTANT_KJ_MOL*float(Tlist[0].split("_")[0]))

    # Get pairwise distances from trajectories
    trajfiles = [ "%s/traj.xtc" % x.rstrip("\n") for x in open("long_temps_last","r").readlines() ]
    native = "%s/Native.pdb" % Tlist[0]
    rij = get_rij(model,trajfiles,native)

    bounds, state_labels = get_state_bounds()
    bounds = [0] + bounds + [model.n_pairs]

    # Found speedups by calculating quantities on per state basis
    U,TS,N,Uframes,TSframes,Nframes = concatenate_state_indicators(Tlist,bounds,coord="Q.dat")

    eps = model.model_param_values[model.fitting_params]

    # Average dimensionless potential energy for each state
    Vp_U  = get_Vp_for_state(model,rij,U,Uframes)
    Vp_TS = get_Vp_for_state(model,rij,TS,TSframes)
    Vp_N  = get_Vp_for_state(model,rij,N,Nframes)

    Vp_U  *= eps
    Vp_TS *= eps
    Vp_N  *= eps

    sumVp_U = np.mean(Vp_U,axis=0)
    sumVp_TS = np.mean(Vp_TS,axis=0)
    sumVp_N = np.mean(Vp_N,axis=0)

    # Correlation matrices
    # <ViVj> - <Vi><Vj>
    corrU  = (np.dot(Vp_U.T,Vp_U)/float(Uframes)) - np.outer(sumVp_U,sumVp_U)
    corrTS = (np.dot(Vp_TS.T,Vp_TS)/float(TSframes)) - np.outer(sumVp_TS,sumVp_TS)
    corrN  = (np.dot(Vp_N.T,Vp_N)/float(Nframes)) - np.outer(sumVp_N,sumVp_N)
    
    varU = np.diag(corrU)
    varTS = np.diag(corrTS)
    varN = np.diag(corrN)

    varTS_to_varU = varTS/varU
    varTS_to_varN = varTS/varN
    varTS2_to_varU_and_varN = (varTS**2)/(varN*varU)
    varTS2_to_varU_or_varN = varTS/(varN + varU)

    contacts = model.pairs[model.fitting_params]

    deps = "\\langle\\delta\\epsilon_{ij}^2\\rangle"
    titles = ["$"+deps+"_{U}$","$"+deps+"_{TS}$","$"+deps+"_{N}$",
              "$"+deps+"_{TS}/"+deps+"_{U}$","$"+deps+"_{TS}/"+deps+"_{N}$",
              "$"+deps+"_{TS}^2/("+deps+"_{U}"+deps+"_{N})$",
              "$"+deps+"_{TS}/("+deps+"_{U} + "+deps+"_{N})$"]

    saveas = ["eps_fluct_U","eps_fluct_TS","eps_fluct_N",
              "eps_fluct_TS_U","eps_fluct_TS_N",
              "eps_fluct_TS_U_and_N","eps_fluct_TS_U_or_N"]

    data_to_plot = [varU, varTS, varN, 
                    varTS_to_varU, varTS_to_varN, 
                    varTS2_to_varU_and_varN, varTS2_to_varU_or_varN]

    if not os.path.exists("plots"):
        os.mkdir("plots")

    if not os.path.exists("fluct"):
        os.mkdir("fluct")
    
    # Save contact maps of:
    #   - U,TS,N fluctuations
    #   - TS fluctuations relative to N or U
    #   - TS fluctuations relative to N and U
    for i in range(len(data_to_plot)):
        C = np.zeros((model.n_residues,model.n_residues))
        data = data_to_plot[i]
        for n in range(len(contacts)):
            C[contacts[n,1] - 1,contacts[n,0] - 1] = data[n]

        plt.figure()
        plt.pcolor(C)
        plt.xlabel("Residue i")
        plt.ylabel("Residue j")
        plt.xlim(0,model.n_residues)
        plt.ylim(0,model.n_residues)
        cbar = plt.colorbar()
        cbar.set_label(titles[i])
        if i > 2:
            plt.title("Relative energy fluctuations  " + titles[i])
        else:
            plt.title("Energy fluctuations  " + titles[i])
        plt.savefig("plots/"+saveas[i]+".png")
        plt.savefig("plots/"+saveas[i]+".pdf")

        np.savetxt("fluct/"+saveas[i]+".dat",data)


    os.chdir("../..")
