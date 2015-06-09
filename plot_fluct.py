import os

import numpy as np
import matplotlib.pyplot as plt

def revert_index(residx,p,n_res):
    if residx <= (n_res - p - 1):
       newidx = residx + p
    else:
       newidx = residx - (n_res - p - 1)
    return newidx

def convert_index(residx,p):
    if residx <= p:
       newidx = residx + 95 - p
    else:
       newidx = residx - p
    return newidx


def plot_fluct_maps(iteration):

    cmap = plt.get_cmap("gnuplot2")
    #cmap.set_bad(color='w',alpha=1.)
    cmap.set_bad(color='gray',alpha=1)

    permutants = [13,33,54,68,81]
    dirs = ["S6"] + [ "cp%d" % x for x in permutants ]

    n_residues = len(open("S6/Native.pdb","r").readlines())
    #iteration = 3
    #emin,emax = np.loadtxt("overall_epsilon_range",unpack=True)
    emin = 0
    emax = 1

    fig,axes = plt.subplots(nrows=2,ncols=3,figsize=(18,9))
    all_Cs = []
    for i in range(len(dirs)):
        os.chdir("%s/iteration_%d" % (dirs[i],iteration))
        Temp = open("long_temps_last" ,"r").readlines()[0].rstrip("\n")
        contacts = np.loadtxt("%s/native_contacts.ndx" % Temp,skiprows=1)
        eps_fluct = np.loadtxt("fluct/eps_fluct_TS_U_or_N.dat")
        eps_fluct = (eps_fluct - eps_fluct.min())/eps_fluct.max()

        col_indx = i % 3
        row_indx = i / 3
        ax = axes[row_indx,col_indx]

        C = np.zeros((n_residues,n_residues),float)*np.nan
        if i == 0:
            for k in range(len(eps_fluct)):
                i_idx = contacts[k,1] 
                j_idx = contacts[k,0]
                C[i_idx-1,j_idx-1] = eps_fluct[k]
        else:
            p = permutants[i-1]
            for k in range(len(eps_fluct)):
                i_idx = contacts[k,1] 
                j_idx = contacts[k,0]
                new_i_idx = revert_index(i_idx,p,n_residues)
                new_j_idx = revert_index(j_idx,p,n_residues)
                if new_j_idx < new_i_idx:
                    C[new_i_idx-1,new_j_idx-1] = eps_fluct[k]
                else:
                    C[new_j_idx-1,new_i_idx-1] = eps_fluct[k]

        all_Cs.append(C)
        C = np.ma.masked_invalid(C)
        image = ax.pcolormesh(C,vmin=emin,vmax=emax,cmap="gnuplot2")
        ax.set_xlim(0,n_residues)
        ax.set_ylim(0,n_residues)
        ax.set_xticks(range(0,n_residues,10))
        ax.set_yticks(range(0,n_residues,10))
        ax.grid(True)
        if row_indx == 0:
            ax.set_xticklabels([])
        if col_indx > 0:
            ax.set_yticklabels([])
        ax.text(60,10,"%s" % dirs[i],fontsize=35,bbox=dict(facecolor='white'))

        #eps_for_ryan = "#%5s%5s%10s\n" % ("i","j","epsilon")
        #for n in range(len(contacts)):
        #    eps_for_ryan += "%5d%5d%10.5f\n" % (contacts[n,0],contacts[n,1],eps[n])
        #open("%s_map_%d_Vanilla" % (dirs[i],iteration),"w").write(eps_for_ryan)
        os.chdir("../..")

    if not os.path.exists("plots"):
        os.mkdir("plots")

    fig.subplots_adjust(right=0.88,wspace=0,hspace=0)
    #fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.9, 0.2, 0.025, 0.6])
    fig.colorbar(image, cax=cbar_ax)
    fig.suptitle("TS Energy Fluctuations Iteration %d" % iteration,fontsize=30)
    #plt.savefig("plots/S6_TS_fluct_%d.png" % iteration)
    #plt.savefig("plots/S6_TS_fluct_%d.pdf" % iteration)
    #plt.savefig("plots/S6_TS_fluct_%d.eps" % iteration)


    plt.show()

if __name__ == "__main__":
    iteration = 2
    plot_fluct_maps(iteration)
