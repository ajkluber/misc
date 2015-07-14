import os
import numpy as np
import matplotlib 
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from misc.cube_cmap import cubecmap

import pymbar

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

def get_ukn(unique_Tlist,sorted_dirs,n_interpolate=5,n_extrapolate=5):
    """Get reduced potential u_kn[k,n] of configuration n at state k""" 

    # Currently assumes temperatures are equally spaced. 
    # TODO: Adaptively add interpolating temperatures.
    Tmin = float(unique_Tlist[0])
    Tmax = float(unique_Tlist[-1])
    dT = float(unique_Tlist[1]) - float(unique_Tlist[0])
    wantdT = dT/float(n_interpolate + 1)
    wantT = np.arange(Tmin,Tmax + wantdT,wantdT)
    extrapolate_below = np.arange(Tmin - dT*n_extrapolate,Tmin,wantdT)
    extrapolate_above = np.arange(Tmax + wantdT, Tmax + dT*n_extrapolate,wantdT)
    wantT = np.concatenate((wantT,extrapolate_below,extrapolate_above))
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
            indices = pymbar.timeseries.subsampleCorrelatedData(tempE,fast=True)
            tempN_k += len(indices)
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
        if i < (len(unique_Tlist) - 1):
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

def compute_heat_capacity_melting_curve(long=False):
    if long:
        dirs = [ x.rstrip("\n") for x in open("long_temps","r").readlines() ]
    else:
        dirs = [ x.rstrip("\n") for x in open("short_temps","r").readlines() ]

    n_contacts = len(open("%s/native_contacts.ndx" % dirs[0],skiprows=1,dtype=int).readlines())

    unique_Tlist, sorted_dirs = get_unique_Tlist(dirs)
    wantT, sub_indices, E, u_kn, N_k = get_ukn(unique_Tlist,sorted_dirs,n_interpolate=10,n_extrapolate=0)
    beta = np.array([ 1./(kb*x) for x in wantT ])
    Tndxs = np.argsort(wantT)

    print "solving mbar"
    mbar = pymbar.MBAR(u_kn,N_k)

    print "computing expectations"
    Q = get_observable("Q.dat",sorted_dirs,sub_indices)
    Q_avg, dQ_avg = mbar.computeExpectations(Q)

    # Energy fluctuations for the heat capacity
    E_avg,dE_avg = mbar.computeExpectations(E)
    E2_avg,dE2_avg = mbar.computeExpectations(E**2)

    # Sort results by increasing temperature. Must do this after mbar or else
    # it breaks.
    wantT = wantT[Tndxs]
    Q_avg = Q_avg[Tndxs] ; dQ_avg = dQ_avg[Tndxs]
    E_avg = E_avg[Tndxs]   ; dE_avg = dE_avg[Tndxs]
    E2_avg = E2_avg[Tndxs] ; dE2_avg = dE2_avg[Tndxs]

    if not os.path.exists("pymbar"):
        os.mkdir("pymbar")
    os.chdir("pymbar")

    Cv = kb*beta*(E2_avg - E_avg**2)

    dQdT = np.array([ abs((Q_avg[i + 1] - Q_avg[i])/(wantT[i + 1] - wantT[i])) for i in range(len(wantT) - 1) ])
    Tmidpoints = np.array([ 0.5*(wantT[i + 1] + wantT[i]) for i in range(len(wantT) - 1) ])

    maxindx = list(Cv).index(max(Cv))
    Tf = wantT[maxindx]
    print "  Folding temperature:   %.2f K" % Tf

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
        deltaT_FWHM = Tmidpoints[rightsearch] - Tmidpoints[leftsearch]
        Omega = ((Tf**2)/(deltaT_FWHM))*maxdQdT
            
        open("omega","w").write("%.2f" % Omega)
        print "  Folding cooperativity: %.2f " % Omega
    except:
        open("omega","w").write("%.2f" % -1)
        print "  couldn't determine folding cooperativity"


    np.savetxt("temps",wantT)
    np.savetxt("cv",Cv)
    np.savetxt("QvsT",Q_avg)

    print "plotting"
    plt.figure()
    plt.plot(wantT,Cv,'r',lw=2)
    plt.title("Heat Capacity")
    plt.savefig("cv.png")
    plt.savefig("cv.pdf")
    plt.savefig("cv.eps")

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    ax1.plot(wantT,Q_avg,'b',lw=2)
    ax1.set_xlabel("Temperature (K)")
    ax1.set_ylabel("$\\left< Q \\right>(T)$")
    ax1.set_ylim(0,n_contacts)
    ax1.set_title("Melting Curve")

    ax2.plot(Tmidpoints,dQdT,'g',lw=2)
    #ax2.set_ylabel("$\\left|\\frac{d\\left< Q \\right>}{dT}\\right|$",rotation="horizontal",fontsize=20)
    ax2.set_ylabel("$\\left|\\frac{d\\left< Q \\right>}{dT}\\right|$",rotation="vertical",fontsize=18)

    plt.savefig("QvsT.png")
    plt.savefig("QvsT.pdf")
    plt.savefig("QvsT.eps")
    #plt.show()

    os.chdir("..")

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
    bin_edges = np.arange(0,165 + 5,5)  # SH3

    #compute_free_energy_profiles_vs_T("Q.dat",0,165,5,"long_temps") 

