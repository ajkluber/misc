import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('--lag', type=int, required=True, help='Lag time used.')
    parser.add_argument('--bins', type=int, required=True, help='Num bins used.')
    args = parser.parse_args()

    lag = args.lag
    bins = args.bins

    os.chdir("lag_time_%d_bins_%d" % (lag,bins))

    exp_tRij = np.loadtxt("expRij.dat")
    w,v = np.linalg.eig(exp_tRij)

    D_all = np.loadtxt("D_all.dat")
    g_all = np.loadtxt("g_all.dat")
    Qbins = np.loadtxt("Qbins.dat")

    plt.figure()
    plt.plot(abs(w),'ro')
    plt.title("Propagator spectrum")

    plt.figure()
    plt.plot(D_all)
    plt.xlabel("Frames")
    plt.ylabel("Diffusion coefficients")

    plt.figure()
    plt.plot(g_all)
    plt.xlabel("Frames")
    plt.ylabel("Free energy")

    plt.figure()
    plt.plot(0.5*(Qbins[1:] + Qbins[:-1]),g_all[-1,:],'b',lw=2)
    plt.xlabel("Q")
    plt.ylabel("F(Q)")

    plt.figure()
    plt.plot(0.5*(Qbins[2:] + Qbins[:-2]),D_all[-1,:],'g',lw=2)
    plt.xlabel("Q")
    plt.ylabel("D(Q)")

    plt.show()
    os.chdir("..")

