
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

def Gaussian_deriv2(r,r0,width):
    return (np.exp(-((r - r0)**2)/(2.*(width**2)))/(width**2))*(1. - ((r - r0)**2)/(width**2))

def get_d2Vp_for_state(model,rij,state,n_frames):
    ''' Get contact energy for subset of frames'''

    time1 = time.time()
    Vp_state = np.zeros((n_frames,model.n_fitting_params),float)
    for i in range(model.n_fitting_params):
        param_idx = model.fitting_params[i]

        # Loop over interactions that use this parameter
        for j in range(len(model.model_param_interactions[param_idx])):
            pair_idx = model.model_param_interactions[param_idx][j]
            
            r0 = model.pairwise_other_parameters[pair_idx][0]
            width = model.pairwise_other_parameters[pair_idx][1]
            Vp_state[:,i] = Vp_state[:,i] + Gaussian_deriv2(rij[state,pair_idx],r0,width)

    time2 = time.time()
    print " Calculating d2Vp took: %.2f sec = %.2f min" % (time2-time1,(time2-time1)/60.)

    return Vp_state

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('--name', type=str, required=True, help='Name of directory.')
    parser.add_argument('--iteration', type=int, default=None, help='Optional. Iteration')
    args = parser.parse_args()

    name = args.name
    if args.iteration is not None:
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
    Forceij_U  = beta*eps*get_dVp_for_state(model,rij,U,Uframes)
    Forceij_TS = beta*eps*get_dVp_for_state(model,rij,TS,TSframes)
    Forceij_N  = beta*eps*get_dVp_for_state(model,rij,N,Nframes)

    d2E_U = ((beta*eps)**2)*get_d2Vp_for_state(model,rij,U,Uframes)
    d2E_TS = ((beta*eps)**2)*get_d2Vp_for_state(model,rij,TS,TSframes)
    d2E_N = ((beta*eps)**2)*get_d2Vp_for_state(model,rij,N,Nframes)

    avgd2E_U = np.mean(d2E_U,axis=0)
    avgd2E_TS = np.mean(d2E_TS,axis=0)
    avgd2E_N = np.mean(d2E_N,axis=0)

    avgForceij_U = np.mean(Forceij_U,axis=0)
    avgForceij_TS = np.mean(Forceij_TS,axis=0)
    avgForceij_N = np.mean(Forceij_N,axis=0)

    HessianU  = -(np.dot(Forceij_U.T,Forceij_U)/float(Uframes)) + np.outer(avgForceij_U,avgForceij_U)
    HessianTS = -(np.dot(Forceij_TS.T,Forceij_TS)/float(TSframes)) + np.outer(avgForceij_TS,avgForceij_TS)
    HessianN  = -(np.dot(Forceij_N.T,Forceij_N)/float(Nframes)) + np.outer(avgForceij_N,avgForceij_N)

    HessianU[np.diag_indices(HessianU.shape[0])] += avgd2E_U
    HessianTS[np.diag_indices(HessianTS.shape[0])] += avgd2E_TS
    HessianN[np.diag_indices(HessianN.shape[0])] += avgd2E_N

    # Hessian should have one unstable mode at the TS, one negative eigenvalue
    egvals_U,  rotation_U = np.linalg.eig(HessianU)
    egvals_TS, rotation_TS = np.linalg.eig(HessianTS)
    egvals_N,  rotation_N = np.linalg.eig(HessianN)


    """
    normal_U = np.dot(Forceij_U,rotation_U)
    normal_TS = np.dot(Forceij_TS,rotation_TS)
    normal_N = np.dot(Forceij_N,rotation_N)

    normal_std_U = np.std(normal_U,axis=0)
    normal_std_TS = np.std(normal_TS,axis=0)
    normal_std_N = np.std(normal_N,axis=0)

    normal_avg_U = np.mean(normal_U,axis=0)
    normal_avg_TS = np.mean(normal_TS,axis=0)
    normal_avg_N = np.mean(normal_N,axis=0)

    varU = np.diag(HessianU)
    varTS = np.diag(HessianTS)
    varN = np.diag(HessianN)
    varU = diag(HessianU)
    varTS = diag(HessianTS)
    varN = diag(HessianN)

    varTS_to_varU = varTS/varU
    varTS_to_varN = varTS/varN
    varTS2_to_varU_and_varN = (varTS**2)/(varN*varU)
    varTS2_to_varU_or_varN = varTS/(varN + varU)

    contacts = model.pairs[model.fitting_params]

    force = "\\langle f_{ij}^2\\rangle"
    titles = ["$"+force+"_{U}$","$"+force+"_{TS}$","$"+force+"_{N}$",
              "$"+force+"_{TS}/"+force+"_{U}$","$"+force+"_{TS}/"+force+"_{N}$",
              "$"+force+"_{TS}^2/("+force+"_{U}"+force+"_{N})$",
              "$"+force+"_{TS}/("+force+"_{U} + "+force+"_{N})$"]

    saveas = ["force_fluct_U","force_fluct_TS","force_fluct_N",
              "force_fluct_TS_U","force_fluct_TS_N",
              "force_fluct_TS_U_and_N","force_fluct_TS_U_or_N"]

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
    C = np.zeros((100,100))
    for n in range(len(pairs)):
        C[pairs[n,1] - 1,pairs[n,0] - 1] = varTS[n]

    plt.figure()
    plt.pcolor(C)
    plt.xlabel("Residue i")
    plt.ylabel("Residue j")
    plt.xlim(0,model.n_residues)
    plt.ylim(0,model.n_residues)
    cbar = plt.colorbar()

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

    np.savetxt("fluct/HessianU.dat",HessianU)
    np.savetxt("fluct/HessianTS.dat",HessianTS)
    np.savetxt("fluct/HessianN.dat",HessianN)
    os.chdir("../..")
    """
