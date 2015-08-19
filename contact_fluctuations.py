import os
import sys
import argparse
import numpy as np

import mdtraj as md

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def get_contact_function(contact_type,continuous):
    acceptable_contact_types = ["LJ1210","Gaussian"]
    if contact_type == "LJ1210":
        if continuous:
            contact_function = lambda r,r0: 0.5*(np.tanh((1.2*r0 - r)/0.5) + 1)
        else:
            contact_function = lambda r,r0: (r <= 1.2*r0).astype(int)
    elif contact_type == "Gaussian":
        if continuous:
            contact_function = lambda r,r0: 0.5*(np.tanh(((r0 + 0.1) - r)/0.5) + 1)
        else:
            contact_function = lambda r,r0: (r <= (r0 + 0.1)).astype(int)
    else:
        raise IOError("--contact_type must be in " + acceptable_contact_types.__str__())
    return contact_function

def get_native_nonnative_contacts(temps_file,contact_type,continuous):
    print "Loading trajectories"
    dirs = [ x.rstrip("\n") for x in open(temps_file,"r").readlines() ]
    trajfiles = [ "%s/traj.xtc" % x for x in dirs ]
    traj = md.load(trajfiles,top="%s/Native.pdb" % dirs[0])

    n_residues = len(open("%s/Native.pdb" % dirs[0],"r").readlines()) - 1
    native_pairs = np.loadtxt("%s/native_contacts.ndx" % dirs[0],skiprows=1,dtype=int) - 1
    n_native_pairs = len(native_pairs)
    r0_native = np.loadtxt("%s/pairwise_params" % dirs[0],usecols=(4,),skiprows=1)[1:2*n_native_pairs:2]
    
    print "Calculating native contacts"
    r_native = md.compute_distances(traj,native_pairs,periodic=False)
    contact_function = get_contact_function(contact_type,continuous)
    Qi_contacts = contact_function(r_native,r0_native)
    Qi_contacts = Qi_contacts.astype(float)
    Q = np.sum(Qi_contacts,axis=1)

    print "Calculating nonnative contacts"
    n_pairwise_lines = len(np.loadtxt("%s/pairwise_params" % dirs[0],usecols=(0,),skiprows=1)[1::2])
    if n_pairwise_lines > n_native_pairs:  
        # Use non-native pairs in pairwise_params file.
        nonnative_pairs = np.loadtxt("%s/pairwise_params" % dirs[0],usecols=(0,1),skiprows=1,dtype=int)[2*n_native_pairs + 1::2] - 1
        r0_nonnative = np.loadtxt("%s/pairwise_params" % dirs[0],usecols=(4,),skiprows=1)[2*n_native_pairs + 1::2]
    else:
        # Construct list of non-native pairs.
        nonnative_pairs = []
        list_native_pairs = [ list(p) for p in native_pairs ]
        for i in range(n_residues):
            for j in range(i + 4,n_residues):
                if [i,j] not in list_native_pairs:
                    nonnative_pairs.append([i,j])
        nonnative_pairs = np.array(nonnative_pairs)
        r0_nonnative = 0.5*np.ones(len(nonnative_pairs),float)
    n_nonnative_pairs = len(nonnative_pairs)
    r_nonnative = md.compute_distances(traj,nonnative_pairs,periodic=False)
    Ai_contacts = contact_function(r_nonnative,r0_nonnative)
    Ai_contacts = Ai_contacts.astype(float)
    A = np.sum(Ai_contacts,axis=1)
    offset = 0
    for i in range(len(dirs)):
        if not os.path.exists("%s/A.dat" % dirs[i]):
            length = file_len("%s/Q.dat" % dirs[i])
            np.savetxt("%s/A.dat" % dirs[i],A[offset:offset + length])
            offset += length

    return n_residues,native_pairs,Qi_contacts,Q,nonnative_pairs,Ai_contacts,A

def plot_contact_probability_map(state_label,n_residues,pairs,contact_probability,coord):
    # Plot contact probabilities
    n_pairs = len(pairs)
    C = np.zeros((n_residues,n_residues))*np.nan
    for p in range(n_pairs):
        C[pairs[p,1], pairs[p,0]] = contact_probability[p]

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
    plt.savefig("map_%s_%s.png" % (coord,state_label),bbox_inches="tight")
    plt.savefig("map_%s_%s.pdf" % (coord,state_label),bbox_inches="tight")
    plt.savefig("map_%s_%s.eps" % (coord,state_label),bbox_inches="tight")

def plot_contact_fluctuations_vs_Q(native_pairs,nonnative_pairs,Qbins,Qi_vs_Q,dQi2_vs_Q,A_vs_Q,Amax_vs_Q,dA2_vs_Q,Ai_vs_Q,dAi2_vs_Q,no_display):
    n_nat = len(native_pairs)
    nat_loops = (native_pairs[:,1] - native_pairs[:,0]).astype(float)
    nat_coloring = [ (nat_loops[i] - min(nat_loops))/(max(nat_loops) - min(nat_loops)) for i in range(n_nat) ]

    n_nnat = len(nonnative_pairs)
    nnat_loops = (nonnative_pairs[:,1] - nonnative_pairs[:,0]).astype(float)
    nnat_coloring = [ (nnat_loops[i] - min(nnat_loops))/(max(nnat_loops) - min(nnat_loops)) for i in range(n_nnat) ]

    plt.figure()
    for i in range(n_nat):
        plt.plot(Qbins,Qi_vs_Q[:,i],color=cubecmap(nat_coloring[i]))
    plt.xlabel("$Q$",fontsize=18)
    plt.ylabel("$\langle Q_i \\rangle$",fontsize=18)
    plt.title("Native contact formation")
    plt.savefig("QivsQ.png",bbox_inches="tight")
    plt.savefig("QivsQ.pdf",bbox_inches="tight")
    plt.savefig("QivsQ.eps",bbox_inches="tight")

    plt.figure()
    for i in range(n_nat):
        plt.plot(Qbins,dQi2_vs_Q[:,i],color=cubecmap(nat_coloring[i]))
    plt.xlabel("$Q$",fontsize=18)
    plt.ylabel("$\langle\delta Q_i^2 \\rangle$",fontsize=18)
    plt.title("Native contact fluctuations")
    plt.savefig("dQi2vsQ.png",bbox_inches="tight")
    plt.savefig("dQi2vsQ.pdf",bbox_inches="tight")
    plt.savefig("dQi2vsQ.eps",bbox_inches="tight")

    plt.figure()
    plt.plot(Qbins,A_vs_Q)
    plt.xlabel("$Q$",fontsize=18)
    plt.ylabel("$\langle A \\rangle$",fontsize=18)
    plt.title("Average non-native contacts")
    plt.savefig("AvsQ.png",bbox_inches="tight")
    plt.savefig("AvsQ.pdf",bbox_inches="tight")
    plt.savefig("AvsQ.eps",bbox_inches="tight")

    plt.figure()
    plt.plot(Qbins,Amax_vs_Q)
    plt.xlabel("$Q$",fontsize=18)
    plt.ylabel("max$\left( A \\right)$",fontsize=18)
    plt.title("Max non-native contacts")
    plt.savefig("AmaxvsQ.png",bbox_inches="tight")
    plt.savefig("AmaxvsQ.pdf",bbox_inches="tight")
    plt.savefig("AmaxvsQ.eps",bbox_inches="tight")

    plt.figure()
    plt.plot(Qbins,dA2_vs_Q)
    plt.xlabel("$Q$",fontsize=18)
    plt.ylabel("$\langle\delta A^2 \\rangle$",fontsize=18)
    plt.title("Non-native contact fluctuations")
    plt.savefig("dA2vsQ.png",bbox_inches="tight")
    plt.savefig("dA2vsQ.pdf",bbox_inches="tight")
    plt.savefig("dA2vsQ.eps",bbox_inches="tight")

    plt.figure()
    for i in range(n_nnat):
        plt.plot(Qbins,Ai_vs_Q[:,i],color=cubecmap(nnat_coloring[i]))
    plt.xlabel("$Q$",fontsize=18)
    plt.ylabel("$\langle A_i \\rangle$",fontsize=18)
    plt.title("Non-native contacts")
    plt.savefig("AivsQ.png",bbox_inches="tight")
    plt.savefig("AivsQ.pdf",bbox_inches="tight")
    plt.savefig("AivsQ.eps",bbox_inches="tight")

    plt.figure()
    for i in range(n_nnat):
        plt.plot(Qbins,dAi2_vs_Q[:,i],color=cubecmap(nnat_coloring[i]))
    plt.xlabel("$Q$",fontsize=18)
    plt.ylabel("$\langle\delta A_i^2 \\rangle$",fontsize=18)
    plt.title("Non-native contact fluctuations")
    plt.savefig("dAi2vsQ.png",bbox_inches="tight")
    plt.savefig("dAi2vsQ.pdf",bbox_inches="tight")
    plt.savefig("dAi2vsQ.eps",bbox_inches="tight")

    if not no_display:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bayesian estimation of 1D diffusion model.")
    parser.add_argument("--temps_file", 
                        type=str, 
                        required=True,
                        help="Name of file with temps to include.")
    parser.add_argument("--contact_type", 
                        type=str,
                        default="Gaussian",
                        help="Use relative cutoff for contact_type contacts. Else assume Gaussian.")
    parser.add_argument("--continuous", 
                        action="store_true",
                        help="Calculate contacts using smooth tanh instead of step function")
    parser.add_argument("--no_display", 
                        action="store_true",
                        help="No access to display, so plots will be saved but not shown.")
    parser.add_argument("--no_plots", 
                        action="store_true",
                        help="Don't plot stuff.")


    # TODO: Allow for alternative folding coordinates besides Q.

    args = parser.parse_args()
    temps_file = args.temps_file
    contact_type = args.contact_type 
    continuous = args.continuous
    no_display = args.no_display
    no_plots = args.no_plots
    n_bins = 30

    if not no_plots:
        if no_display:
            import matplotlib
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from misc.cube_cmap import cubecmap

    n_residues,native_pairs,Qi_contacts,Q,nonnative_pairs,Ai_contacts,A = get_native_nonnative_contacts(temps_file,contact_type,continuous)
    n_native_pairs = len(native_pairs)
    n_nonnative_pairs = len(nonnative_pairs)

    if not os.path.exists("contact_fluct_vs_Q"):
        os.mkdir("contact_fluct_vs_Q")
    os.chdir("contact_fluct_vs_Q")
    if not no_plots:
        if not os.path.exists("plots"):
            os.mkdir("plots")

    ###########################################################################
    # Calculate contact formation 
    ###########################################################################
    if os.path.exists("../state_bounds.txt"):
        state_labels = []
        state_bounds = []
        for line in open("../state_bounds.txt","r"):
            state_labels.append(line.split()[0])
            state_bounds.append([int(line.split()[1]),int(line.split()[2])])

        for i in range(len(state_labels)):
            if not os.path.exists("cont_prob_%s.dat" % state_labels[i]):
                if i == 0:
                    print "Calculating contact probability for:"
                print "  state %s" % state_labels[i]
                state_indicator = (Q > state_bounds[i][0]) & (Q < state_bounds[i][1])
                Qi_for_state = np.mean(Qi_contacts[state_indicator,:],axis=0)
                Ai_for_state = np.mean(Ai_contacts[state_indicator,:],axis=0)
                np.savetxt("Ai_%s.dat" % state_labels[i],Ai_for_state)
                if not no_plots:
                    os.chdir("plots")
                    plot_contact_probability_map(state_labels[i],n_residues,native_pairs,Qi_for_state,"Qi")
                    plot_contact_probability_map(state_labels[i],n_residues,nonnative_pairs,Ai_for_state,"Ai")
                    os.chdir("..")
                np.savetxt("Qi_%s.dat" % state_labels[i],Qi_for_state)

    ###########################################################################
    # Calculate contact fluctuations vs reaction coordinate
    ###########################################################################
    print "Calculating Qi with %d" % n_bins
    Qi_vs_Q = np.zeros((n_bins,n_native_pairs),float)
    dQi2_vs_Q = np.zeros((n_bins,n_native_pairs),float)
    A_vs_Q = np.zeros(n_bins,float)
    dA2_vs_Q = np.zeros(n_bins,float)
    Amax_vs_Q = np.zeros(n_bins,float)
    Ai_vs_Q = np.zeros((n_bins,n_nonnative_pairs),float)
    dAi2_vs_Q = np.zeros((n_bins,n_nonnative_pairs),float)
    minQ = min(Q)
    maxQ = max(Q)
    Qbins = np.linspace(minQ,maxQ,n_bins)
    incQ = (float(maxQ) - float(minQ))/float(n_bins)
    for n in range(n_bins):
        state_indicator = (Q > (minQ + n*incQ)) & (Q <= (minQ + (n+1)*incQ))
        Qi_vs_Q[n,:] = np.mean(Qi_contacts[state_indicator,:],axis=0)
        dQi2_vs_Q[n,:] = np.var(Qi_contacts[state_indicator,:],axis=0)
        A_vs_Q[n] = np.mean(A[state_indicator])
        dA2_vs_Q[n] = np.var(A[state_indicator])
        Amax_vs_Q[n] = np.max(A[state_indicator])
        Ai_vs_Q[n,:] = np.mean(Ai_contacts[state_indicator,:],axis=0)
        dAi2_vs_Q[n,:] = np.var(Ai_contacts[state_indicator,:],axis=0)
    
    np.savetxt("Qbins.dat",Qbins)
    np.savetxt("QivsQ.dat",Qi_vs_Q)
    np.savetxt("dQi2vsQ.dat",dQi2_vs_Q)
    np.savetxt("AvsQ.dat",A_vs_Q)
    np.savetxt("dA2vsQ.dat",dA2_vs_Q)
    np.savetxt("AmaxvsQ.dat",Amax_vs_Q)
    np.savetxt("AivsQ.dat",Ai_vs_Q)
    np.savetxt("dAi2vsQ.dat",dAi2_vs_Q)

    if not no_plots:
        os.chdir("plots")
        plot_contact_fluctuations_vs_Q(native_pairs,nonnative_pairs,Qbins,Qi_vs_Q,dQi2_vs_Q,A_vs_Q,Amax_vs_Q,dA2_vs_Q,Ai_vs_Q,dAi2_vs_Q,no_display)
        os.chdir("..")
    os.chdir("..")