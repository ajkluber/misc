import os
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt

import pyemma.coordinates as coor

# TODO: Make complementary plotting script for:
#   - PMF of TICA 1
#   - PMF of TICA 1 vs Q.
#   - PMF of TICA 1 vs TICA 2 (if possible).
#   - TICA eigenvalues
# Take correlation with Q to determine which way is 'folded'
# Q = np.hstack([ np.loadtxt("%s/Q.dat" % x) for x in dirs])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('--temps', type=str, required=True, help='File holding directory names.')
    parser.add_argument('--lag', type=int, required=True, help='Lag to use for TICA.')
    parser.add_argument('--stride', type=int, required=True, help='Stride to use for TICA.')
    parser.add_argument('--feature', type=str, required=True, help='Input feature to TICA.')
    #parser.add_argument('--start', type=int, default=0,help='Starting index.')
    #parser.add_argument('--verbose', actions='store_true', help='Verbose.')
    args = parser.parse_args()

    tempsfile = args.temps
    lag = args.lag
    stride = args.stride
    feature = args.feature
    
    available_features = ["native_contacts","all_contacts"]
    if feature not in available_features:
        raise IOError("--feature should be in: %s" % available_features.__str__())

    logging.basicConfig(filename="tica_%d_%d.log" % (lag,stride),
                        filemode="w",
                        format="%(levelname)s:%(name)s:%(asctime)s: %(message)s",
                        datefmt="%H:%M:%S",
                        level=logging.DEBUG)

    temps = [ x.rstrip("\n") for x in open(tempsfile,"r").readlines() ]
    uniq_Tlist = []
    Qlist = []
    Tlist = []
    for i in range(len(temps)):
        T = temps[i].split("_")[0]
        if T not in uniq_Tlist:
            uniq_Tlist.append(T)
            Tlist.append([temps[i]])
        else:
            idx = uniq_Tlist.index(T)
            Tlist[idx].append(temps[i])

    logging.info("TICA inputs")
    logging.info("  lag       = %d" % lag)
    logging.info("  stride    = %d" % stride)

    # For each unique temperature. Run TICA
    for i in range(len(Tlist)):
        dirs = Tlist[i]
        logging.info("Running TICA T = %s" % uniq_Tlist[i])

        traj_list = [ "%s/traj.xtc" % x for x in dirs ]
        topfile = "%s/Native.pdb" % dirs[0]
        n_residues = len(open(topfile,"r").readlines()) - 1

        logging.info("  picking features: ")
        if feature == "all_contacts":
            logging.info("    contacts between all pairs")
            pairs = []
            for n in range(n_residues):
                for m in range(n + 4,n_residues):
                    pairs.append([n,m])
            pairs = np.array(pairs)
        else:
            logging.info("    contacts between native pairs")
            pairs = np.loadtxt("%s/native_contacts.ndx" % dirs[0],dtype=int,skiprows=1) - 1

        # Featurizer parameterizes a pipeline to read in trajectory in chunks.
        feat = coor.featurizer(topfile)
        feat.add_contacts(pairs,threshold=0.8,periodic=False)

        # Source trajectories
        logging.info("  sourcing trajectories: %s" % traj_list.__str__())
        inp = coor.source(traj_list, feat)

        # Stride has a drastic influence on the number of acceptable eigenvalues.
        logging.info("  computing TICA")
        tica_obj = coor.tica(inp, lag=lag, stride=stride, var_cutoff=0.9, kinetic_map=True)

        # Check if eigenvalues go negative at some point. Truncate before that if necessary.
        first_neg_eigval = np.where(tica_obj.eigenvalues < 0)[0][0]
        logging.info("  TICA done. Index of first negative eigenvalue: %d" % first_neg_eigval)
        keep_dims = min([tica_obj.dimension(),first_neg_eigval])

        # Save principal TICA coordinate in each subdirectory
        logging.info("  getting output from TICA object")
        Y = tica_obj.get_output(dimensions=np.arange(1)) # get tica coordinates

        contact_weights = np.vstack((pairs[:,0],pairs[:,1],tica_obj.eigenvectors[:,0])).T

        logging.info("  saving contact weights")
        for n in range(len(dirs)):
            os.chdir(dirs[n])
            np.savetxt("tica1_weights.dat",contact_weights)
            np.savetxt("tica1_%d_%d.dat" % (lag,stride),Y[n][:,0])
            os.chdir("..")

