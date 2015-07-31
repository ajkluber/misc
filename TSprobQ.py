import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mdtraj as md

def plot_contact_probability_map(state_label,n_residues,native_pairs,contact_probability):
    # Plot contact probabilities
    n_native_pairs = len(native_pairs)
    C = np.zeros((n_residues,n_residues))*np.nan
    for p in range(n_native_pairs):
        C[native_pairs[p,1], native_pairs[p,0]] = contact_probability[p]

    plt.figure()
    cmap = plt.get_cmap("Blues")
    cmap.set_bad(color="lightgray",alpha=1.)
    C = np.ma.masked_invalid(C)
    plt.pcolormesh(C,vmin=0,vmax=1,cmap=cmap)
    plt.title("%s contact probablility" % state_label,fontsize=15)
    plt.xlabel("Residue i",fontsize=16)
    plt.ylabel("Residue j",fontsize=16)
    cbar = plt.colorbar()
    cbar.set_label("Contact probability")
    plt.xlim(0,n_residues)
    plt.ylim(0,n_residues)
    plt.xticks(range(0,n_residues,10))
    plt.yticks(range(0,n_residues,10))
    plt.grid(True)
    plt.savefig("plots/cont_prob_%s.png" % state_label,bbox_inches="tight")
    plt.savefig("plots/cont_prob_%s.pdf" % state_label,bbox_inches="tight")
    plt.savefig("plots/cont_prob_%s.eps" % state_label,bbox_inches="tight")

if __name__ == "__main__":
    temps = sys.argv[1]
    dirs = [ x.rstrip("\n") for x in open(temps,"r").readlines() ]
    trajfiles = [ "%s/traj.xtc" % x for x in dirs ]
    gaussian_contacts = True

    if not os.path.exists("plots"):
        os.mkdir("plots")

    if os.path.exists("early_conts"):
        early = np.loadtxt("early_conts",dtype=int)
        late = np.loadtxt("late_conts",dtype=int)
        calcqearly = True
    else:
        calcqearly = False

    state_labels = []
    state_bounds = []
    for line in open("state_bounds.txt","r"):
        state_labels.append(line.split()[0])
        state_bounds.append([int(line.split()[1]),int(line.split()[2])])

    n_residues = len(open("%s/Native.pdb" % dirs[0],"r").readlines()) - 1
    native_pairs = np.loadtxt("%s/native_contacts.ndx" % dirs[0],skiprows=1,dtype=int) - 1
    n_native_pairs = len(native_pairs)
    native_distances = np.loadtxt("%s/pairwise_params" % dirs[0],usecols=(4,),skiprows=1)[1::2*n_native_pairs]
    print "loading trajectories"
    traj = md.load(trajfiles,top="%s/Native.pdb" % dirs[0])
    for i in range(len(state_labels)):
        print "calculating state", state_labels[i]
        state_label = state_labels[i]
        state_bound = state_bounds[i]
        n_frames = 0
        distances = md.compute_distances(traj,native_pairs,periodic=False)
        if gaussian_contacts:
            contacts = (distances <= (native_distances + 0.1)).astype(int)
        else:
            contacts = (distances <= 1.2*native_distances).astype(int)
        contacts = contacts.astype(float)
        Q = np.sum(contacts,axis=1)
        if i == 0:
            contact_probability = np.zeros(n_native_pairs,float) 

        state_indicator = (Q > state_bound[0]) & (Q < state_bound[1])
        n_frames += float(sum(state_indicator))
        contact_probability += np.sum(contacts[state_indicator == True,:],axis=0)
        contact_probability /= n_frames
        np.savetxt("cont_prob_%s.dat" % state_label,contact_probability)

        plot_contact_probability_map(state_label,n_residues,native_pairs,contact_probability)

        if calcqearly:
            qearly = np.sum(contacts[:,early],axis=1)
            qlate = np.sum(contacts[:,late],axis=1)
            np.savetxt("qearly.dat",qearly)
            np.savetxt("qlate.dat",qlate)
