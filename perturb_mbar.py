import os
import time
import numpy as np
import matplotlib 
matplotlib.use("Agg")

import matplotlib.pyplot as plt

import model_builder as mbd
from misc.cube_cmap import cubecmap

import pymbar
import mdtraj as md
import model_builder as mdb
#from lsdmap.lsdmap.rw import coord_reader

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

    # Currently assumes temperatures are equally spaced. 
    # TODO: Adaptively add interpolating temperatures.
    Tmin = float(unique_Tlist[0])
    Tmax = float(unique_Tlist[-1])
    dT = float(unique_Tlist[1]) - float(unique_Tlist[0])
    wantdT = dT/float(n_interpolate + 1)
    wantT0 = np.arange(Tmin,Tmax + wantdT,wantdT)
    extrapolate_below = np.arange(Tmin - wantdT*n_extrapolate,Tmin,wantdT)
    extrapolate_above = np.arange(Tmax + wantdT, Tmax + wantdT*n_extrapolate,wantdT)
    wantT = np.concatenate((wantT0,extrapolate_below,extrapolate_above))
    beta = np.array([ 1./(kb*x) for x in wantT ])

    sub_indices = []
    N_k = []
    first = True
    print "loading E"
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

        # Pad zeros for interpolating between simulation temps.
        if i < (len(sorted_dirs) - 1):
            for j in range(n_interpolate):
                N_k.append(0)

    # Pad zeros for extrapolating beyond sampled data. 
    for i in range(len(extrapolate_below) + len(extrapolate_above)):
        N_k.append(0)
    N_k = np.array(N_k)
    u_kn = np.zeros((len(wantT),len(E)))

    print "calculating u_kn"
    for i in range(len(wantT)):
        u_kn[i,:] = beta[i]*E

    return wantT, sub_indices, E, u_kn, N_k

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


def compute_free_energy_profiles_vs_T(coordinate,coordmin,coordmax,dcoord,tempsfile):
    """Compute potential of mean force along a coordinate as a function of temperature"""

    bin_edges = np.arange(coordmin,coordmax + dcoord,dcoord)

    bin_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])
    
    dirs = [ x.rstrip("\n") for x in open(tempsfile,"r").readlines() ]

    unique_Tlist, sorted_dirs = get_unique_Tlist(dirs)
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
    #compute_heat_capacity_melting_curve()
    
    #bin_edges = np.arange(0,260 + 10,10) # S6
    #bin_edges = np.arange(0,165 + 5,5)  # SH3

#def perturb_heat_capacity_melting_curve():

    #model,fitopts = mdb.inputs.load_model("SH3_go") 
    #model,fitopts = mdb.inputs.load_model("PDZ") 
    model,fitopts = mdb.inputs.load_model("1E0G") 
    n_pairs = model.n_native_pairs
    pair_V = model.pair_V[1::2]
    pairs = model.pairs[1:2*n_pairs:2] - 1
    eps0 = model.model_param_values[1::2]

    os.chdir("1E0G/iteration_0")

    dirs = [ x.rstrip("\n") for x in open("pert_temps","r").readlines() ]
     
    unique_Tlist, sorted_dirs = get_unique_Tlist(dirs)
    #wantT, sub_indices, E, u_kn, N_k = get_ukn(unique_Tlist,sorted_dirs,n_interpolate=5,n_extrapolate=3,stride=10)
    wantT, sub_indices, E, u_kn, N_k = get_ukn(unique_Tlist,sorted_dirs,n_interpolate=5,n_extrapolate=2,stride=10)
    beta = np.array([ 1./(kb*x) for x in wantT ])

    print "Loading distances"
    distances = np.zeros((len(E),len(pairs)),float)
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

    print "solving mbar"
    mbar = pymbar.MBAR(u_kn,N_k,subsampling=1)

    # Compute Cv and melting curve of a perturbed Hamiltonian at all temperatures.
    Tndxs = np.argsort(wantT)
    variance = 0.01
    Q = get_observable("Q.dat",sorted_dirs,sub_indices)

    for m in range(10): 
        print "perturbation: ", m

        E_prime = np.array(E,copy=True)
        deps = np.random.normal(loc=0,scale=np.sqrt(variance),size=len(eps0))
        deps -= np.mean(deps)
        deps *= np.sqrt(variance)/np.std(deps) 
        dE = np.zeros(distances.shape,float)
        for p in range(n_pairs):
            dE[:,p] = deps[p]*pair_V[p](distances[:,p])
        E_prime += np.sum(dE,axis=1)

        # u_kn_prime = Perturbed energy at many new temperatures
        u_kn_prime = np.zeros(u_kn.shape,float)
        #E_prime = E + np.random.normal(loc=0,scale=0.1*np.mean(E),size=len(E))
        for i in range(len(beta)):
            u_kn_prime[i,:] = beta[i]*E_prime

        print "perturbing observables"
        Q_avg, dQ_avg = mbar.computeExpectations(Q,u_kn=u_kn_prime)

        # Energy fluctuations for the heat capacity
        E_avg,dE_avg = mbar.computeExpectations(E_prime,u_kn=u_kn_prime)
        E2_avg,dE2_avg = mbar.computeExpectations(E_prime**2,u_kn=u_kn_prime)
        Cv = kb*beta*(E2_avg - E_avg**2)

        plt.figure(1)
        plt.plot(wantT[Tndxs],Cv[Tndxs])

        plt.figure(2)
        plt.plot(wantT[Tndxs],Q_avg[Tndxs])

    if not os.path.exists("pymbar_pert"):
        os.mkdir("pymbar_pert")

    os.chdir("pymbar_pert")

    #np.savetxt("temps",wantT[Tndxs])
    #np.savetxt("cv",Cv[Tndxs])
    #np.savetxt("QvsT",Q_avg[Tndxs])

    print "saving"
    #plt.figure(1)
    #plt.plot(wantT[Tndxs],Cv[Tndxs],'r',lw=2)
    #plt.title("Heat Capacity")
    #plt.savefig("cv.png")
    #plt.savefig("cv.pdf")
    #plt.savefig("cv.eps")

    #plt.figure(2)
    #plt.plot(wantT[Tndxs],Q_avg[Tndxs],'b',lw=2)
    #plt.title("Melting Curve")
    #plt.savefig("QvsT.png")
    #plt.savefig("QvsT.pdf")
    #plt.savefig("QvsT.eps")
    #plt.show()

    Q_avg, dQ_avg = mbar.computeExpectations(Q)

    # Energy fluctuations for the heat capacity
    E_avg,dE_avg = mbar.computeExpectations(E)
    E2_avg,dE2_avg = mbar.computeExpectations(E**2)
    Cv = kb*beta*(E2_avg - E_avg**2)
    
    plt.figure(1)
    plt.plot(wantT[Tndxs],Cv[Tndxs],'k',lw=2)
    plt.title("Heat Capacity")
    plt.savefig("cv.png")
    plt.savefig("cv.pdf")
    plt.savefig("cv.eps")

    plt.figure(2)
    plt.plot(wantT[Tndxs],Q_avg[Tndxs],'k',lw=2)
    plt.title("Melting Curve")
    plt.savefig("QvsT.png")
    plt.savefig("QvsT.pdf")
    plt.savefig("QvsT.eps")

    os.chdir("..")
    os.chdir("../..")
