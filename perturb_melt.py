import os
import time
import logging
import argparse
import numpy as np
import matplotlib 
matplotlib.use("Agg")

import matplotlib.pyplot as plt

import model_builder as mbd
from misc.cube_cmap import cubecmap

import mdtraj as md
import pymbar
import model_builder as mdb

kb = 0.0083145

def get_unique_Tlist(dirs):
    """Get unique list of temperatures and sorted directory list"""
    unique_Tlist = []
    sorted_dirs = []
    for i in range(len(dirs)):
        tempT = dirs[i].split("_")[0]
        if tempT not in unique_Tlist: 
            unique_Tlist.append(tempT)
            sorted_dirs.append([])

        index = unique_Tlist.index(tempT)
        sorted_dirs[index].append(dirs[i])
    return unique_Tlist,sorted_dirs

def get_ukn(unique_Tlist,sorted_dirs,n_interpolate=5,n_extrapolate=5,stride=1):
    """Get reduced potential u_kn[k,n] of configuration n at state k""" 

    sub_indices = []
    N_k = []
    wantT = []
    first = True
    for i in range(len(sorted_dirs)):
        # Runs at same state (temperature).
        tempN_k = 0
        temp_indices = []
        for j in range(len(sorted_dirs[i])):
            os.chdir("%s" % sorted_dirs[i][j])
            tempE = np.loadtxt("energyterms.xvg",usecols=(5,))
            if stride == 1:
                indices = pymbar.timeseries.subsampleCorrelatedData(tempE)
            else:
                indices = range(0,len(tempE),stride)
            tempN_k += len(indices)
            #print "%s" % sorted_dirs[i][j], len(tempE), tempE[0], tempE[-1]
            temp_indices.append(indices)
            if first:
                E = tempE[indices]
                first = False
            else:
                E = np.concatenate((E,tempE[indices]))
            os.chdir("..")
         
        N_k.append(tempN_k)

        # Save frame indices for loading observables.
        sub_indices.append(temp_indices)

        wantT.append(float(unique_Tlist[i]))

        # Pad zeros for interpolating between simulation temps.
        if i < (len(sorted_dirs) - 1):
            T0 = float(unique_Tlist[i])
            T1 = float(unique_Tlist[i + 1])
            for j in range(n_interpolate):
                wantT.append(T0 + float(j + 1)*(T1 - T0)/float(n_interpolate + 1))
                N_k.append(0)
    
    # Pad zeros for extrapolating beyond sampled data. 
    for i in range(n_extrapolate):
        wantT.append(float(unique_Tlist[0]) - float(i + 1)*(T1 - T0))
        N_k.append(0)
    for i in range(n_extrapolate):
        wantT.append(float(unique_Tlist[-1]) + float(i + 1)*(T1 - T0))
        N_k.append(0)
    wantT = np.array(wantT)
    beta = np.array([ 1./(kb*x) for x in wantT ])
    N_k = np.array(N_k)
    u_kn = np.zeros((len(wantT),len(E)))

    for i in range(len(wantT)):
        u_kn[i,:] = beta[i]*E

    return wantT, sub_indices, E, u_kn, N_k

def get_distances(sorted_dirs,sub_indices,n_frames,pairs):
    """Get pairwise distances """
    distances = np.zeros((n_frames,len(pairs)),float)
    frame_num = 0
    frame_start = 0
    for i in range(len(sorted_dirs)):
        for j in range(len(sorted_dirs[i])):
            os.chdir(sorted_dirs[i][j])
            starttime = time.time()
            traj = md.load("traj.xtc",top="Native.pdb")
            N = len(sub_indices[i][j])
            distances[frame_start:frame_start + N,:] = md.compute_distances(traj,pairs,periodic=False)[sub_indices[i][j],:]

            frame_start += N
            print " %.4f sec" % (time.time() - starttime)
            os.chdir("..")
    return distances

def calculate_contacts(model,distances):
    """NOT DONE"""
    contacts = np.zeros(distances.shape,float)
    pair_r0 = np.array([ model.pairwise_other_params[i][0] for i in range(1,len(model.pairwise_other_params),2) ])


def get_observable(filename,sorted_dirs,sub_indices):
    """Get observable and sub-sample by statistical inefficiency"""
    first = True
    for i in range(len(sorted_dirs)):
        # Collect data at same state (temperature).
        tempN_k = 0
        temp_indices = []
        for j in range(len(sorted_dirs[i])):
            os.chdir("%s" % sorted_dirs[i][j])
            tempA = np.loadtxt(filename)[sub_indices[i][j]]
            if first:
                A = tempA
                first = False
            else:
                A = np.concatenate((A,tempA))
            os.chdir("..")
    return A

def get_binned_observables(bin_edges,filename,sorted_dirs,sub_indices):
    """Get observable and sub-sample by statistical inefficiency"""
    A_indicator = [ None for x in range(len(bin_edges) - 1) ]
    first = True
    for i in range(len(sorted_dirs)):
        # Collect data at same state (temperature).
        tempN_k = 0
        temp_indices = []
        for j in range(len(sorted_dirs[i])):
            os.chdir("%s" % sorted_dirs[i][j])
            tempA = np.loadtxt(filename)[sub_indices[i][j]]
            if first:
                for n in range(len(bin_edges) - 1):
                    temp_indicator = (tempA > bin_edges[n]).astype(int)*(tempA <= bin_edges[n + 1]).astype(int)
                    A_indicator[n] = temp_indicator
                first = False
            else:
                for n in range(len(bin_edges) - 1):
                    temp_indicator = (tempA > bin_edges[n]).astype(int)*(tempA <= bin_edges[n + 1]).astype(int)
                    A_indicator[n] = np.concatenate((A_indicator[n],temp_indicator))
            os.chdir("..")

    A_indicator = np.array(A_indicator)
    return A_indicator

def calculate_Tf_cooperativity(T,Cv,Q_avg):
    maxindx = list(Cv).index(max(Cv))
    Tf = T[maxindx]

    dQdT = np.array([ abs((Q_avg[i+1] - Q_avg[i])/(T[i+1] - T[i])) for i in range(len(T) - 1) ])
    Tmid = np.array([ 0.5*(T[i+1] + T[i]) for i in range(len(T) - 1) ])
    maxindx_Q = list(dQdT).index(max(dQdT))
    Tf_Q = Tmid[maxindx_Q]

    try:
        maxdQdT = dQdT[maxindx - 1]
        rightsearch = maxindx - 1
        for i in range(len(dQdT)):
            if dQdT[rightsearch] <= (maxdQdT/2.):
                break
            else:
                rightsearch += 1
        leftsearch = maxindx - 1
        for i in range(len(dQdT)):
            if dQdT[leftsearch] <= (maxdQdT/2.):
                break
            else:
                leftsearch -= 1
        deltaT_FWHM = T[rightsearch] - T[leftsearch]
        omega = ((Tf**2)/(deltaT_FWHM))*maxdQdT
    except:
        omega = -1
    return Tf, omega, Tmid, dQdT

def compute_free_energy_profiles_vs_T(coordinate,coordmin,coordmax,dcoord,tempsfile):
    """Compute potential of mean force along a coordinate as a function of temperature"""

    bin_edges = np.arange(coordmin,coordmax + dcoord,dcoord)

    bin_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])
    
    dirs = [ x.rstrip("\n") for x in open(tempsfile,"r").readlines() ]

    unique_Tlist, sorted_dirs = get_unique_Tlist(dirs)
    print "Loading u_kn"
    wantT, sub_indices, E, u_kn, N_k = get_ukn(unique_Tlist,sorted_dirs)
    beta = np.array([ 1./(kb*x) for x in wantT ])

    print "solving mbar"
    mbar = pymbar.MBAR(u_kn,N_k)

    # Compute bin expectations at all temperatures. Free energy profile.
    print "computing expectations"
    A_indicators = get_binned_observables(bin_edges,coordinate,sorted_dirs,sub_indices)

    coordname = coordinate.split(".")[0]
    if not os.path.exists("pymbar_%s" % coordname):
        os.mkdir("pymbar_%s" % coordname)
    os.chdir("pymbar_%s" % coordname)

    Tndxs = np.argsort(wantT)
    sortT = wantT[Tndxs]

    np.savetxt("outputT",wantT[Tndxs])
    np.savetxt("bin_edges",bin_edges)
    np.savetxt("bin_centers",bin_centers)

    plt.figure()
    # Compute pmf at each temperature.
    for i in range(len(wantT)):
        Tstr = "%.2f" % wantT[Tndxs[i]]
        #x = (max(sortT) - sortT[i])/(max(sortT) - min(sortT))   # Reverse color ordering
        x = (sortT[i] - min(sortT))/(max(sortT) - min(sortT))   # Color from blue to red
        
        A_ind_avg, dA_ind_avg, d2A_ind_avg = mbar.computeMultipleExpectations(A_indicators,u_kn[Tndxs[i],:])

        pmf = -np.log(A_ind_avg)
        min_pmf = min(pmf)
        pmf -= min_pmf
        pmf_err_lower = -np.log(A_ind_avg - dA_ind_avg) - min_pmf
        pmf_err_upper = -np.log(A_ind_avg + dA_ind_avg) - min_pmf
        np.savetxt("free_%s" % Tstr, np.array([pmf,pmf_err_lower,pmf_err_upper]).T)
    
        plt.plot(bin_centers,pmf,lw=2,label=Tstr,color=cubecmap(x))

    plt.xlabel("%s" % coordname,fontsize=18)
    plt.ylabel("F(%s)" % coordname,fontsize=18)
    
    plt.savefig("Fvs%s_all.png" % coordname)
    plt.savefig("Fvs%s_all.pdf" % coordname)
    plt.savefig("Fvs%s_all.eps" % coordname)

    os.chdir("..")

if __name__ == "__main__":

    #native_heterogeneity = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    #native_variance = [ x**2 for x in native_heterogeneity ]

    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('--name', type=str, required=True, help='Name of directory.')
    parser.add_argument('--native_std', type=float, required=True, help='Native heterogeneity std. dev.')
    parser.add_argument('--n_pert', type=int, default=10, help='Number of perturbations.')
    args = parser.parse_args()

    name = args.name
    native_std = args.native_std
    n_pert = args.n_pert

    if not os.path.exists("%s/iteration_0/pymbar_pert_%.2f" % (name,native_std)):
        os.mkdir("%s/iteration_0/pymbar_pert_%.2f" % (name,native_std))

    logging.basicConfig(filename='%s/iteration_0/pymbar_pert_%.2f/perturb_melt.log' % (name,native_std),
                        filemode='w',
                        format="%(levelname)s:%(name)s:%(asctime)s: %(message)s",
                        datefmt="%H:%M:%S",
                        level=logging.DEBUG)

    ####################################################################
    # Load model, solve MBAR, load observable
    ####################################################################
    model,fitopts = mdb.inputs.load_model(name) 
    n_pairs = model.n_native_pairs
    pair_V = model.pair_V[1::2]
    pairs = model.pairs[1:2*n_pairs:2] - 1
    eps0 = model.model_param_values[1::2]

    os.chdir("%s/iteration_0" % name)
     
    dirs = [ x.rstrip("\n") for x in open("pert_temps","r").readlines() ]
    starttime = time.time()
    unique_Tlist, sorted_dirs = get_unique_Tlist(dirs)
    logging.info("loading u_kn")
    wantT, sub_indices, E, u_kn, N_k = get_ukn(unique_Tlist,sorted_dirs,n_interpolate=5,n_extrapolate=3,stride=10)
    beta = np.array([ 1./(kb*x) for x in wantT ])

    logging.info("calculating distances")
    distances = get_distances(sorted_dirs,sub_indices,len(E),pairs) 

    logging.info("solving mbar")
    mbar = pymbar.MBAR(u_kn,N_k,subsampling=1)

    logging.info("loading Q")
    Q = get_observable("Q.dat",sorted_dirs,sub_indices)
    
    ####################################################################
    # Perturb Hamiltonian then compute Cv, QvsT, and cooperativity
    ####################################################################
    #pymbar_perturbation():
    Tndxs = np.argsort(wantT)
    deps_all = np.zeros((n_pert,n_pairs),float)
    Cv_all = np.zeros((n_pert,len(wantT)),float)
    Qavg_all = np.zeros((n_pert,len(wantT)),float)
    dQdTavg_all = np.zeros((n_pert,len(wantT) - 1),float)
    Tf_all = np.zeros(n_pert,float)
    omega_all = np.zeros(n_pert,float)
    for m in range(n_pert): 
        starttime = time.time()

        E_prime = np.array(E,copy=True)
        deps = np.random.normal(loc=0,scale=native_std,size=len(eps0))
        deps -= np.mean(deps)
        deps *= native_std/np.std(deps) 
        deps_all[m,:] = deps
        dE = np.zeros(distances.shape,float)
        for p in range(n_pairs):
            dE[:,p] = deps[p]*pair_V[p](distances[:,p])
        E_prime += np.sum(dE,axis=1)

        # u_kn_prime = Perturbed energy at many new temperatures
        u_kn_prime = np.zeros(u_kn.shape,float)
        #E_prime = E + np.random.normal(loc=0,scale=0.1*np.mean(E),size=len(E))
        for i in range(len(beta)):
            u_kn_prime[i,:] = beta[i]*E_prime

        Q_avg, dQ_avg = mbar.computeExpectations(Q,u_kn=u_kn_prime)

        # Energy fluctuations for the heat capacity
        E_avg,dE_avg = mbar.computeExpectations(E_prime,u_kn=u_kn_prime)
        E2_avg,dE2_avg = mbar.computeExpectations(E_prime**2,u_kn=u_kn_prime)
        Cv = kb*beta*(E2_avg - E_avg**2)

        # Calculate cooperative and Tf
        Tf, omega, Tmid, dQdT = calculate_Tf_cooperativity(wantT[Tndxs],Cv[Tndxs],Q_avg[Tndxs]/float(model.n_native_pairs))
        
        Tf_all[m] = Tf
        omega_all[m] = omega
        Cv_all[m,:] = Cv[Tndxs]
        Qavg_all[m,:] = Q_avg[Tndxs]
        dQdTavg_all[m,:] = dQdT

        plt.figure(1)
        plt.plot(wantT[Tndxs],Cv[Tndxs],'b')

        plt.figure(2)
        plt.plot(wantT[Tndxs],Q_avg[Tndxs],'b')

        plt.figure(3)
        plt.plot(Tmid,dQdT,'b')

        logging.info("  perturbation: " + str(m) + " %.4f sec" % (time.time() - starttime))

    os.chdir("pymbar_pert_%.2f" % native_std)

    logging.info("saving files")
    np.savetxt("outputT",wantT[Tndxs])
    np.savetxt("outputTmid",Tmid)
    np.savetxt("Tf_all",Tf_all)
    np.savetxt("omega_all",omega_all)
    np.savetxt("Cv_all",Cv_all)
    np.savetxt("Qavg_all",Qavg_all)


    # Energy fluctuations and melting curve of unperturbed Hamiltonian.
    E_avg,dE_avg = mbar.computeExpectations(E)
    E2_avg,dE2_avg = mbar.computeExpectations(E**2)
    Cv = kb*beta*(E2_avg - E_avg**2)
    Q_avg, dQ_avg = mbar.computeExpectations(Q)
    Tf, omega, Tmid, dQdT = calculate_Tf_cooperativity(wantT[Tndxs],Cv[Tndxs],Q_avg[Tndxs]/float(model.n_native_pairs))

    open("unpert_Tf","w").write("%.2f" % Tf)
    open("unpert_omega","w").write("%.2f" % omega)
    np.savetxt("unpert_Cv",Cv[Tndxs])
    np.savetxt("unpert_Qavg",Q_avg[Tndxs])
    
    logging.info("saving plots")
    plt.figure(1)
    plt.plot(wantT[Tndxs],Cv[Tndxs],'k',lw=3)
    plt.title("Perturbed Heat Capacity %s $\\sigma = %.2f$" % (name,native_std))
    plt.ylabel("Cv")
    plt.xlabel("T")
    plt.savefig("cv.png")
    plt.savefig("cv.pdf")
    plt.savefig("cv.eps")

    plt.figure(2)
    plt.plot(wantT[Tndxs],Q_avg[Tndxs],'k',lw=3)
    plt.title("Melting Curve %s $\\sigma = %.2f$" % (name,native_std))
    plt.ylabel("Q")
    plt.xlabel("T")
    plt.savefig("QvsT.png")
    plt.savefig("QvsT.pdf")
    plt.savefig("QvsT.eps")

    plt.figure(3)
    plt.plot(Tmid,dQdT,'k',lw=3)
    plt.title("Melting Curve Slope %s $\\sigma = %.2f$" % (name,native_std))
    plt.ylabel("dQ/dT")
    plt.xlabel("T")
    plt.savefig("dQdT.png")
    plt.savefig("dQdT.pdf")
    plt.savefig("dQdT.eps")

    os.chdir("..")
    os.chdir("../..")
    logging.info("done")
