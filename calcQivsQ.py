import argparse
import os
import numpy as np

def get_contact_probability_versus_Q(temps_file="long_temps_last",n_bins=30):
    ''' Calculate the contact probabilities versus Q '''
    if not os.path.exists("QivsQ.dat"):
        print "  calculating Qi vs Q"
        n_frames = 0.
        temps = [ x.rstrip("\n") for x in open(temps_file, "r").readlines() ]
        for i in range(len(temps)):
            T = temps[i]
            Q_temp = np.loadtxt("%s/Q.dat" % T)
            Qi_temp = np.loadtxt("%s/qimap.dat" % T)
            if i == 0:
                Qi = Qi_temp
                Q = Q_temp
            else:
                Qi = np.concatenate((Qi,Qi_temp),axis=0)
                Q = np.concatenate((Q,Q_temp),axis=0)

        counts = np.zeros(n_bins)
        Qi_vs_bins = np.zeros((n_bins,len(Qi[0,:])),float)
        minQ = min(Q)
        maxQ = max(Q)
        incQ = (float(maxQ) - float(minQ))/float(n_bins)

        print "  sorting Qi"
        for i in range(len(Q)):
            for n in range(n_bins):
                if ((minQ + n*incQ) < Q[i]) and (Q[i] < (minQ + (n+1)*incQ)):
                    Qi_vs_bins[n,:] += Qi[i,:]
                    counts[n] += 1.

        Qi_vs_Q = (Qi_vs_bins.T/counts).T
        Qbins = np.linspace(minQ,maxQ,n_bins)
        np.savetxt("QivsQ.dat",Qi_vs_Q)
        np.savetxt("Qbins.dat",Qbins)
    else:
        print "  loading Qi vs Q"
        Qi_vs_Q = np.loadtxt("QivsQ.dat")
        Qbins = np.loadtxt("Qbins.dat")
    return Qbins, Qi_vs_Q


def route_measure(Qbins,Qi_vs_Q,n_bins):
    Q = Qbins/float(max(Qbins))
    route = np.zeros(n_bins)
    for i in range(n_bins):
        if (Q[i] == 0) or (Q[i] == 1):
            pass
        else:
            route[i] = (1./(Q[i]*(1. - Q[i])))*(np.std(Qi_vs_Q[i,:])**2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('--temps', type=str, default="long_temps_last", help='Name of file that hold temps')
    parser.add_argument('--n_bins', type=int, default=30, help='Name of file that hold temps')
    args = parser.parse_args()

    temps_file = args.temps
    n_bins = args.n_bins

    get_contact_probability_versus_Q(temps_file=temps_file,n_bins=n_bins)
