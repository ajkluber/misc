
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
    states = [U,TS,N]
    state_frames = [Uframes,TSframes,Nframes]

    eps = model.model_param_values[model.fitting_params]

    #Eij  = beta*eps*get_Vp_for_state(model,rij,np.ones(rij.shape[0],bool),rij.shape[0])
    #normal_TS_full = np.dot(Eij,rotation_TS)

    # Average dimensionless potential energy for each state
    Eij_U  = beta*eps*get_Vp_for_state(model,rij,U,Uframes)
    Eij_TS = beta*eps*get_Vp_for_state(model,rij,TS,TSframes)
    Eij_N  = beta*eps*get_Vp_for_state(model,rij,N,Nframes)

    avgEij_U = np.mean(Eij_U,axis=0)
    avgEij_TS = np.mean(Eij_TS,axis=0)
    avgEij_N = np.mean(Eij_N,axis=0)

    # Eigenvectors of the local energy covariance matrices are 
    # ''normal coordinates'' in which perturbation models becomes 
    # more or less exact.
    # C_ij = [<E_i E_j> - <E_i><E_j>]
    corrU  = (np.dot(Eij_U.T,Eij_U)/float(Uframes)) - np.outer(avgEij_U,avgEij_U)
    corrTS = (np.dot(Eij_TS.T,Eij_TS)/float(TSframes)) - np.outer(avgEij_TS,avgEij_TS)
    corrN  = (np.dot(Eij_N.T,Eij_N)/float(Nframes)) - np.outer(avgEij_N,avgEij_N)
    
    egvals_U,  rotation_U = np.linalg.eig(corrU)
    egvals_TS, rotation_TS = np.linalg.eig(corrTS)
    egvals_N,  rotation_N = np.linalg.eig(corrN)

    normal_U = np.dot(Eij_U,rotation_U)
    normal_TS = np.dot(Eij_TS,rotation_TS)
    normal_N = np.dot(Eij_N,rotation_N)

    normal_std_U = np.std(normal_U,axis=0)
    normal_std_TS = np.std(normal_TS,axis=0)
    normal_std_N = np.std(normal_N,axis=0)

    normal_avg_U = np.mean(normal_U,axis=0)
    normal_avg_TS = np.mean(normal_TS,axis=0)
    normal_avg_N = np.mean(normal_N,axis=0)

    rotations  = [rotation_U,rotation_TS,rotation_N]
    normal_std = [normal_std_U,normal_std_TS,normal_std_N]
    normal_avg = [normal_avg_U,normal_avg_TS,normal_avg_N]

    if not os.path.exists("FEPestimate"):
        os.mkdir("FEPestimate")
    os.chdir("FEPestimate")

    alpha = 1.0

    """
    plt.figure()
    colors = ['r','b','g']
    for i in range(3):
    #for i in range(1):
        bin_frames = states[i]
        n_frames = float(state_frames[i])

        # Dimensionless average energy and standard deviation
        avg_Eij = np.sum(Eij[bin_frames,:],axis=0)/n_frames
        std_Eij = np.std(Eij[bin_frames,:],axis=0)

        #############################################################
        # Comparison data: FEP calculation from simulation
        #############################################################
        dG_FEP  = -np.log(np.sum(np.exp(alpha*Eij[bin_frames,:]),axis=0)/n_frames)

        #dG_FEP_normal_U  = -np.log(np.sum(np.exp(alpha*normal_U),axis=0)/float(Uframes))
        #dG_FEP_normal_TS  = -np.log(np.sum(np.exp(alpha*normal_TS),axis=0)/float(TSframes))
        #dG_FEP_normal_N  = -np.log(np.sum(np.exp(alpha*normal_N),axis=0)/float(Nframes))

        #dG_FEP_U  = -np.log(np.sum(np.exp(alpha*Eij[U,:]),axis=0)/float(Uframes))
        #dG_FEP_TS  = -np.log(np.sum(np.exp(alpha*Eij[TS,:]),axis=0)/float(TSframes))
        #dG_FEP_N  = -np.log(np.sum(np.exp(alpha*Eij[N,:]),axis=0)/float(Nframes))

        #############################################################
        # Estimate 1: Gaussian distribution in contact basis
        #############################################################
        dG_estimate_1 = 0.5*(alpha**2)*(std_Eij**2) - alpha*avg_Eij

        #dG_estimate_normal_U = 0.5*(alpha**2)*(normal_std_U**2) - alpha*normal_avg_U
        #dG_estimate_normal_TS = 0.5*(alpha**2)*(normal_std_TS**2) - alpha*normal_avg_TS
        #dG_estimate_normal_N = 0.5*(alpha**2)*(normal_std_N**2) - alpha*normal_avg_N

        #############################################################
        # Estimate 2: Gaussian distribution in normal basis
        #############################################################
        dG_estimate_2 = 0.5*(alpha**2)*np.dot(rotations[i]**2,normal_std[i]**2) - alpha*np.dot(rotations[i],normal_avg[i])
        #dG_estimate_3 = 0.5*(alpha**2)*normal_std[i]**2 - alpha*normal_avg[i]

        #dG_estimate_U = 0.5*(alpha**2)*np.dot(rotation_U**2,normal_std_U**2) - alpha*np.dot(rotation_U,normal_avg_U)
        #dG_estimate_TS = 0.5*(alpha**2)*np.dot(rotation_TS**2,normal_std_TS**2) - alpha*np.dot(rotation_TS,normal_avg_TS)
        #dG_estimate_N = 0.5*(alpha**2)*np.dot(rotation_N**2,normal_std_N**2) - alpha*np.dot(rotation_N,normal_avg_N)

        plt.plot(dG_estimate_1,dG_FEP,'r.')
        plt.plot(dG_estimate_2,dG_FEP,'b.')
        #plt.plot(dG_estimate_normal,dG_FEP_normal,'g.')

    maxdG = max([max(dG_estimate_1),max(dG_estimate_2),max(dG_FEP)])

    plt.plot([0],[0],'r.',label="Contact Basis")
    plt.plot([0],[0],'b.',label="Normal Basis")
    plt.plot([0,maxdG],[0,maxdG],'k')
    plt.legend(loc=2)
    plt.xlabel("Model estimates $\\Delta G$ (k$_B$T)")
    plt.ylabel("Full FEP $\\Delta G$ (k$_B$T)")
    plt.title("Full FEP vs Gaussian models. Perturbation $\\alpha$ = %.2f" % alpha)
    plt.savefig("dG_Gauss_vs_full_%.2f.pdf" % alpha)
    plt.savefig("dG_Gauss_vs_full_%.2f.png" % alpha)

    plt.show()

    """

    os.chdir("..")
    os.chdir("../..")
