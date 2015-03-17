""" Script to run MBAR calculation


Got heat capacity and melting curve calculation to work fine.


Trying to get temperature dependent ddG's.



"""

import pickle
import argparse
import os
import shutil
import numpy as np
import subprocess as sb
#import matplotlib.pyplot as plt

#from pymbar import MBAR

import model_builder as mdb
import project_tools as pjt

global R_KJ_MOL 
R_KJ_MOL = 0.0083145

def jeff_WHAM_ddG(mutants):

    # Collect simulation energies and Q.
    
    #dirs = [ x.rstrip("\n") for x in open("long_temps","r").readlines() ]
    #sim_temps = np.array([ float(x.split("_")[0]) for x in dirs ])
    sim_temps = np.unique(np.array([ float(x.split("_")[0]) for x in open("long_temps","r").readlines() ]))
    E = np.array([ np.concatenate((np.loadtxt("%.2f_1/energyterms.xvg" % x,usecols=(5,)),
                                            np.loadtxt("%.2f_2/energyterms.xvg" % x,usecols=(5,)),
                                            np.loadtxt("%.2f_3/energyterms.xvg" % x,usecols=(5,)))) for x in sim_temps ])


    if not os.path.exists("wham_ddG"):
        os.mkdir("wham_ddG")

    # Loop over mutations
    for k in range(len(mutants)):
    #for k in [0]:

        mut = mutants[k]
        print "  loading dHk_%s" % mut
        dH_U = np.array([ np.concatenate((np.loadtxt("%.2f_1/dHk_%s_U.dat" % (x,mut)),
                                         np.loadtxt("%.2f_2/dHk_%s_U.dat" % (x,mut)),
                                         np.loadtxt("%.2f_3/dHk_%s_U.dat" % (x,mut)))) for x in sim_temps ])

        dH_TS = np.array([ np.concatenate((np.loadtxt("%.2f_1/dHk_%s_TS.dat" % (x,mut)),
                                         np.loadtxt("%.2f_2/dHk_%s_TS.dat" % (x,mut)),
                                         np.loadtxt("%.2f_3/dHk_%s_TS.dat" % (x,mut)))) for x in sim_temps ])

        dH_N = np.array([ np.concatenate((np.loadtxt("%.2f_1/dHk_%s_N.dat" % (x,mut)),
                                         np.loadtxt("%.2f_2/dHk_%s_N.dat" % (x,mut)),
                                         np.loadtxt("%.2f_3/dHk_%s_N.dat" % (x,mut)))) for x in sim_temps ])

        os.chdir("wham_ddG")
        for i in range(len(sim_temps)):
            # Reaction coordinate is an indicator function
            indicator = np.zeros((3,len(dH_N[i]))) 
            indicator[i,dH_U[i] != 0] = 1
            indicator[i,dH_TS[i] != 0] = 2
            indicator[i,dH_N[i] != 0] = 3
            dH = dH_U[i,:] + dH_TS[i,:] + dH_N[i,:]

            temp_histogram = np.vstack((E[i,:],indicator[i,:],dH)).T
            np.savetxt("hist.%.2f" % sim_temps[i],temp_histogram,fmt="%10.5f %10.5f %10.5f")

        print "  running wham"
        # Run wham and rename results directory
        if not os.path.exists("free"):
            os.mkdir("free")
        sb.call("java -jar WHAM.jar --config config",shell=True)
        shutil.move("free","%s" % mut)
        os.chdir("..")

    os.chdir("..")

def mbar_ddG(mutants):

    # Collect simulation energies and Q.
    
    #dirs = [ x.rstrip("\n") for x in open("long_temps","r").readlines() ]
    #sim_temps = np.array([ float(x.split("_")[0]) for x in dirs ])
    sim_temps = np.unique(np.array([ float(x.split("_")[0]) for x in open("long_temps","r").readlines() ]))
    E_from_sim = np.array([ np.concatenate((np.loadtxt("%.2f_1/energyterms.xvg" % x,usecols=(5,)),
                                            np.loadtxt("%.2f_2/energyterms.xvg" % x,usecols=(5,)),
                                            np.loadtxt("%.2f_3/energyterms.xvg" % x,usecols=(5,)))) for x in sim_temps ])

    n_out_points = 10
    n_sims = len(E_from_sim)
    n_samps = len(E_from_sim[0,:])

    #inter_temps = np.linspace(min(sim_temps) - 0.05*np.mean(sim_temps),max(sim_temps) + 0.05*np.mean(sim_temps),n_out_points)
    inter_temps = np.linspace(min(sim_temps),max(sim_temps),n_out_points)
    out_temps = np.concatenate((sim_temps,inter_temps))
    beta_k = 1./(R_KJ_MOL*out_temps)
    K = len(out_temps)

    E_kn = np.zeros((K,n_samps),float)
    N_k = np.zeros(K,int)

    E_kn[:n_sims,:] = E_from_sim
    N_k[:n_sims] = n_samps*np.ones(n_sims,int)

    print " calculating u_kln"
    u_kln = np.zeros((K,K,n_samps),float)
    E_kln = np.zeros((K,K,n_samps),float)
    for k in range(K):
        for l in range(K):
            u_kln[k,l,:] = beta_k[l]*E_kn[k,:]
            E_kln[k,l,:] = E_kn[k,:]

    if not os.path.exists("mbar_ddG"):
        os.mkdir("mbar_ddG")

    print "  initializing mbar"
    if not os.path.exists("mbar_ddG/mbar_fk"):
        mbar = MBAR(u_kln,N_k,verbose=True)
        np.savetxt("mbar_ddG/mbar_fk",mbar.f_k)
    else:
        f_k = np.loadtxt("mbar_ddG/mbar_fk")
        mbar = MBAR(u_kln,N_k,verbose=True,initial_f_k=f_k)

    np.savetxt("mbar_ddG/temps",out_temps[n_sims:])

    # Loop over mutations
    for k in range(len(mutants)):

        mut = mutants[k]
        print "  loading dHk_%s" % mut
        for state in ["U","TS","N"]:
            dH_from_sim = np.array([ np.concatenate((np.loadtxt("%.2f_1/dHk_%s_%s.dat" % (x,mut,state)),
                                                     np.loadtxt("%.2f_2/dHk_%s_%s.dat" % (x,mut,state)),
                                                     np.loadtxt("%.2f_3/dHk_%s_%s.dat" % (x,mut,state)))) for x in sim_temps ])

            h_indicator_from_sim = np.array([ (x != 0).astype(int).astype(float) for x in dH_from_sim ])

            # Compute dH 
            dH_kn = np.zeros((K,n_samps),float)
            dH_kn[:n_sims,:] = dH_from_sim
            h_indicator = np.zeros((K,n_samps),float)
            h_indicator[:n_sims,:] = h_indicator_from_sim

            A_kln = np.zeros((K,K,n_samps),float)
            for k in range(K):
                for l in range(K):
                    A_kln[k,l,:] = h_indicator[k,:]*np.exp(-beta_k[l]*dH_kn[k,:])

            print "    computing expdH vs T "
            expdH_expect, dexpdH_expect = mbar.computeExpectations(A_kln)

            print "    saving expdH"
            os.chdir("mbar_ddG")
            np.savetxt("expdH_%s_%s" % (mut,state),np.vstack((expdH_expect[n_sims:],dexpdH_expect[n_sims:])).T)
            os.chdir("..")

    os.chdir("..")


def mbar_heat_capacity_melting_curve():

    # Collect simulation energies and Q.
    dirs = [ x.rstrip("\n") for x in open("short_temps_last","r").readlines() ]
    sim_temps = np.array([ float(x.split("_")[0]) for x in dirs ])
    E_from_sim = np.array([ np.loadtxt("%s/energyterms.xvg" % x,usecols=(5,)) for x in dirs ])
    Qkn = np.array([ np.loadtxt("%s/Q.dat" % x) for x in dirs ])
    n_out_points = 100
    n_sims = len(E_from_sim)
    n_samps = len(E_from_sim[0,:])

    inter_temps = np.linspace(min(sim_temps),max(sim_temps),n_out_points)
    out_temps = np.concatenate((sim_temps,inter_temps))
    beta_k = 1./(R_KJ_MOL*out_temps)
    K = len(out_temps)

    E_kn = np.zeros((K,n_samps),float)
    A_kn = np.zeros((K,n_samps),float)
    N_k = np.zeros(K,int)

    E_kn[:n_sims,:] = E_from_sim
    A_kn[:n_sims,:] = Qkn
    N_k[:n_sims] = n_samps*np.ones(n_sims,int)

    u_kln = np.zeros((K,K,n_samps),float)
    E_kln = np.zeros((K,K,n_samps),float)
    for k in range(K):
        for l in range(K):
            u_kln[k,l,:] = beta_k[l]*E_kn[k,:]
            E_kln[k,l,:] = E_kn[k,:]

    if not os.path.exists("mbar"):
        os.mkdir("mbar")
    os.chdir("mbar")

    if not os.path.exists("mbar_fk"):
        print "  initializing mbar"
        mbar = MBAR(u_kln,N_k,verbose=True)
        np.savetxt("mbar_fk",mbar.f_k)
    else:
        f_k = np.loadtxt("mbar_fk")
        mbar = MBAR(u_kln,N_k,verbose=True,initial_f_k=f_k)

    print "  computing heat capacity"
    E_expect, dE_expect = mbar.computeExpectations(E_kln)
    E2_expect,dE2_expect = mbar.computeExpectations(E_kln**2)
    Cv = (E2_expect - (E_expect*E_expect))/(R_KJ_MOL*out_temps**2)

    Tf_indx = list(Cv[n_sims:]).index(max(Cv[n_sims:]))

    if (Tf_indx == 0) or (Tf_indx == len(Cv[n_sims:])):
        print " warning: Tf_indx is an endpoint. Probably didn't find the peak"

    Tf = out_temps[n_sims + Tf_indx]
    print "  Estimated folding temperature %.2f" % Tf

    print "  computing melting curve"
    Q_expect, dQ_expect = mbar.computeExpectations(A_kn)

    np.savetxt("temps",out_temps[n_sims:])
    np.savetxt("Q_vs_T",np.vstack((Q_expect[n_sims:],dQ_expect[n_sims:])).T)
    np.savetxt("cv",np.vstack((Cv[n_sims:],Cv[n_sims:])).T)

    os.chdir("..")


if __name__ == "__main__":
    #parser = argparse.ArgumentParser(description='.')
    #parser.add_argument('--name', type=str, required=True, help='Name of subdirectory.')
    #parser.add_argument('--iteration', type=int, required=True, help='Iteration.')
    #args = parser.parse_args()
    #name = args.name
    #iteration = args.iteration
    
    name = "S6"
    iteration = 2
    
    os.chdir("%s/mutants" % name)
    mutants = pjt.parameter_fitting.ddG_MC2004.mutatepdbs.get_all_core_mutations()
    os.chdir("../..")
    
    os.chdir("%s/iteration_%d" % (name,iteration))
    
    #mbar_heat_capacity_melting_curve()
    #mbar_ddG(mutants)

    jeff_WHAM_ddG(mutants)

    os.chdir("../..")
