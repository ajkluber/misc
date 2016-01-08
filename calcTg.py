import numpy as np
import matplotlib.pyplot as plt
import os

def get_energy_IS(filename, size, U_bounds):
    # Load inherent structure energies
    allE = [ np.loadtxt("rank_{}/{}".format(rank, filename)) for rank in range(size) ]
    frame_idxs = [ np.loadtxt("rank_{}/frame_idxs.dat".format(rank), dtype=int) for rank in range(size) ]
    n_finish = [ len(x) for x in allE ]
    frames_fin = [ frame_idxs[i][:n_finish[i]] for i in range(size) ]
    Q = np.loadtxt("../Qtanh_0_05.dat")
    U = [ ((Q[x] > U_bounds[0]) & (Q[x] < U_bounds[1])) for x in frames_fin ]
    E = np.concatenate(allE)
    E_U = np.concatenate([ allE[i][U[i]] for i in range(size) ])
    return E, E_U

if __name__ == "__main__":
    kb = 0.0083145
    beta = 1./(127.30*kb)

    size = 12
    bounds = (25,35)
    bounds = (0,35)
    nbins = 100
    nbins_U = 30
    Etot, Etot_U = get_energy_IS("Etot.dat", size, bounds) 
    Enn, Enn_U = get_energy_IS("Enonnative.dat", size, bounds) 

    # Density of state analysis
    nE, bins = np.histogram(Etot, bins=nbins)
    dE = bins[1] - bins[0]
    probE = nE.astype(float)*dE
    probE[probE == 0] = np.min(probE[probE != 0])
    mid_bin = 0.5*(bins[1:] + bins[:-1])
    minE = np.min(mid_bin)
    mid_bin -= minE

    # Compute Tg using Etot
    #nE_U, bins_U = np.histogram(Etot_U, bins=nbins_U, density=True)
    # OR Compute Tg using Enonnative
    nE_U, bins_U = np.histogram(Enn_U, bins=nbins_U, density=True)
    prob_U = np.float(len(Etot_U))/np.float(len(Etot))
    dE_U = bins_U[1] - bins_U[0]
    probE_U = nE_U.astype(float)*dE_U
    probE_U[probE_U == 0] = np.min(probE_U[probE_U != 0])
    mid_bin_U = 0.5*(bins_U[1:] + bins_U[:-1])
    mid_bin_U -= minE

    omegaE = (probE/probE[0])*np.exp(beta*mid_bin)
    SconfE = np.log(omegaE)

    omegaE_U = ((prob_U*probE_U)/(probE[0]))*np.exp(beta*mid_bin_U)
    SconfE_U = np.log(omegaE_U)

    # REM fit to the configurational entropy yields Tg!
    coeff = np.polyfit(mid_bin_U, SconfE_U, 2)
    a, b, c = coeff
    E_GS = (-b + np.sqrt(b*b - 4.*a*c))/(2.*a)
    SconfE_U_interp = np.poly1d(coeff)
    dSdE = np.poly1d(np.array([2.*a, b]))
    Tg = 1./(kb*dSdE(E_GS))
    print Tg
    # Tg is ~74K using Etot or ~2.5K using Enonnative

    # solve for other REM parameters using fit.

    # Plot
    plt.figure()
    plt.plot(mid_bin, SconfE, label="$S_{conf}$")
    plt.plot(mid_bin, beta*mid_bin, label="$E$")
    plt.plot(mid_bin_U, SconfE_U, label="$S_{conf}(Q_u)$")
    plt.plot(mid_bin, SconfE_U_interp(mid_bin), label="REM fit")
    plt.ylabel("Entropy $S_{conv}$")
    plt.xlabel("Energy")
    plt.title("Determining glass temperature")
    plt.legend(loc=2)
    plt.ylim(0, np.max(SconfE))
    plt.savefig("REM_fit_of_dos.png", bbox_inches="tight") 
    plt.savefig("REM_fit_of_dos.pdf", bbox_inches="tight") 
    plt.show()



