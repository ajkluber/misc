import os
import sys
import numpy as np

import mdtraj as md


if __name__ == "__main__":
    temps = sys.argv[1]
    dirs = [ x.rstrip("\n") for x in open(temps,"r").readlines() ]
    gaussian_contacts = True

    if os.path.exists("early_conts"):
        early = np.loadtxt("early_conts",dtype=int)
        late = np.loadtxt("late_conts",dtype=int)
        calcqearly = True
    else:
        calcqearly = False

    raise IOError("This isn't done yet!!!")

    state_labels = []
    state_bounds = []
    for line in open("state_bounds.txt","r"):
        state_labels.append(line.split()[0])
        state_bounds.append([int(line.split()[1]),int(line.split()[2])])

    native_pairs = np.loadtxt("%s/native_contacts.ndx" % dirs[0],skiprows=1,dtype=int) - 1

    # Get nonnative pairs
    n_native_pairs = len(native_pairs)
    nonnative_pairs = np.loadtxt("%s/pairwise_params" % dirs[0],usecols=(0,1),skiprows=1)[2*n_native_pairs::2]
    nonnative_distances = np.loadtxt("%s/pairwise_params" % dirs[0],usecols=(4,),skiprows=1)[2*n_native_pairs::2]

    for i in range(len(state_labels)):
        print "calculating state", state_labels[i]
        state_label = state_labels[i]
        state_bound = state_bounds[i]
        n_frames = 0
        for j in range(len(dirs)):
            os.chdir(dirs[j])
            traj = md.load("traj.xtc",top="Native.pdb")
            distances = md.compute_distances(traj,native_pairs,periodic=False)
            if gaussian_contacts:
                contacts = (distances <= (native_distances + 0.1)).astype(int)
            else:
                contacts = (distances <= 1.2*native_distances).astype(int)
            contacts = contacts.astype(float)
            Q = np.sum(contacts,axis=1)
            if i == 0:
                contact_probability = np.zeros(n_native_pairs,float) 

            state_indicator = ((Q > state_bound[0]).astype(int)*(Q < state_bound[1]).astype(int)).astype(bool)
            n_frames += float(sum(state_indicator))
            contact_probability += np.sum(contacts[(state_indicator == True),:],axis=0)

            #if calcqearly:
            #    qearly = np.sum(qimap[:,early],axis=1)
            #    qlate = np.sum(qimap[:,late],axis=1)
            #    np.savetxt("qearly.dat",qearly)
            #    np.savetxt("qlate.dat",qlate)

            os.chdir("..")
        contact_probability /= n_frames
        np.savetxt("cont_prob_%s.dat" % state_label,contact_probability)
