import os
import argparse
import shutil
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('--temps', type=str, required=True, help='File holding directory names.')
    parser.add_argument('--lag', type=int, required=True, help='Lag to use for TICA.')
    parser.add_argument('--stride', type=int, required=True, help='Stride to use for TICA.')
    parser.add_argument('--feature', type=str, required=True, help='Input feature to TICA.')
    args = parser.parse_args()

    tempsfile = args.temps
    lag = args.lag
    stride = args.stride
    feature = args.feature

    available_features = ["native_contacts","all_contacts"]
    if feature not in available_features:
        raise IOError("--feature should be in: %s" % available_features.__str__())

    if feature == "all_contacts":
        prefix = "all"
    else:
        prefix = "nat"

    temps = [ x.rstrip("\n") for x in open(tempsfile,"r").readlines() ]

    for i in range(len(temps)):
        os.chdir(temps[i])
        psi1 = np.loadtxt("tica1_%s_%d.dat" % (prefix,lag))
        if i == 0:
            Q = np.loadtxt("Q.dat")
            corr = np.sign(np.dot(Q,psi1)/(np.linalg.norm(psi1)*np.linalg.norm(Q)))
        if corr == -1:
            print "changing sign of psi1"
            np.savetxt("tica1_%s_%d.dat" % (prefix,lag),corr*psi1)
        os.chdir("..")

    if corr == -1:
        psi1_w = np.loadtxt("tica_%s_%d/tica1_weights.dat" % (prefix,lag))
        psi1_w[:,2] *= -1
        np.savetxt("tica_%s_%d/tica1_weights.dat" % (prefix,lag),psi1_w)
         
