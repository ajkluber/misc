import numpy as np
import matplotlib.pyplot as plt

import pyemma.coordinates as coor
#import pyemma.msm as msm
#import pyemma.plots as mplt

# TODO: Make complementary plotting script for:
#   - PMF of TICA 1
#   - PMF of TICA 1 vs Q.
#   - PMF of TICA 1 vs TICA 2 (if possible).
#   - TICA eigenvalues


if __name__ == "__main__":

    # Data resides here
    dirs = ["137.60_%d" % x for x in range(1,4) ]
    traj_list = [ "%s/traj.xtc" % x for x in dirs ]
    topfile = "%s/Native.pdb" % dirs[0]
    n_residues = len(open(topfile,"r").readlines()) - 1
    native_pairs = np.loadtxt("%s/native_contacts.ndx" % dirs[0],dtype=int,skiprows=1) - 1

    # NOTE: from pylab import * was used here. Need to add numpy,plt namespaces.

    all_pairs = []
    for i in range(n_residues):
        for j in range(i + 4,n_residues):
            all_pairs.append([i,j])
    all_pairs = np.array(all_pairs)

    # Featurizer
    print "setting up featurizer"
    feat = coor.featurizer(topfile)
    #feat.add_backbone_torsions(cossin=True)
    #feat.add_distances_ca(periodic=False)
    #feat.add_contacts(all_pairs,threshold=0.8,periodic=False)
    feat.add_contacts(native_pairs,threshold=0.8,periodic=False)

    # Source trajectories
    print 'sourcing trajectories'
    inp = coor.source(traj_list, feat)

    # Stride has a drastic influence on the number of acceptable eigenvalues.
    tica_obj = coor.tica(inp, lag=100, stride=5, var_cutoff=0.9, kinetic_map=True)

    # Check if eigenvalues go negative at some point. Truncate before that if necessary.
    first_neg_eigval = np.where(tica_obj.eigenvalues < 0)[0][0]
    print first_neg_eigval

    #hist(tica_obj.eigenvectors[:,0],bins=30)
    Q = hstack([ loadtxt("137.60_%d/Q.dat" % x) for x in [1,2,3]])

    # Save principal TICA coordinate in each subdirectory
    Y = tica_obj.get_output(dimensions=arange(1)) # get tica coordinates

    # Take correlation with Q to determine which way is 'folded'

