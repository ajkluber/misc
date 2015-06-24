import numpy as np
import os
import sys

if __name__ == "__main__":
    temps = sys.argv[1]
    dirs = [ x.rstrip("\n") for x in open(temps,"r").readlines() ]

    state_labels = []
    state_bounds = []
    for line in open("state_bounds.txt","r"):
        state_labels.append(line.split()[0])
        state_bounds.append([int(line.split()[1]),int(line.split()[2])])

    for i in range(len(state_labels)):
        state_label = state_labels[i]
        state_bound = state_bounds[i]
        n_frames = 0
        for i in range(len(dirs)):
            os.chdir(dirs[i])
            Q = np.loadtxt("Q.dat")
            qimap = np.loadtxt("qimap.dat")
            if i == 0:
                contact_probability = np.zeros(qimap.shape[1],float) 

            state_indicator = ((Q > state_bound[0]).astype(int)*(Q < state_bound[1]).astype(int)).astype(bool)
            n_frames += float(sum(state_indicator))
            contact_probability += np.sum(qimap[(state_indicator == True),:],axis=0)

            os.chdir("..")
        contact_probability /= n_frames
        np.savetxt("cont_prob_%s.dat" % state_label,contact_probability)

