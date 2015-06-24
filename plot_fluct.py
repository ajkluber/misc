import os

import numpy as np
import matplotlib.pyplot as plt

from misc.cube_cmap import cubecmap,cubeYFcmap

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

    #cmap = plt.get_cmap("gnuplot2")
    cmap = cubecmap

    cmap.set_bad(color='w',alpha=1.)
    #cmap.set_bad(color='gray',alpha=1)

    permutants = [13,33,54,68,81]
    dirs = ["S6"] + [ "cp%d" % x for x in permutants ]

    n_residues = len(open("S6/Native.pdb","r").readlines())
    #iteration = 3
    #emin,emax = np.loadtxt("overall_epsilon_range",unpack=True)
    emin = 0
    emax = 1

    fig = plt.figure(figsize=(18,9))
    ax0 = fig.add_subplot(111)
    axes = np.array([[fig.add_subplot(231),fig.add_subplot(232),fig.add_subplot(233)],
                     [fig.add_subplot(234),fig.add_subplot(235),fig.add_subplot(236)]])
    #fig,axes = plt.subplots(nrows=2,ncols=3,figsize=(18,9))

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
        #C = np.ma.masked_invalid(C)
        C = np.ma.array(C,mask=np.isnan(C))
        image = ax.pcolormesh(C,vmin=emin,vmax=emax,cmap=cmap)
        ax.set_xlim(0,n_residues)
        ax.set_ylim(0,n_residues)
        #ax.set_aspect('equal', 'datalim')
        ax.set_xticks(range(0,n_residues,10))
        ax.set_yticks(range(0,n_residues,10))
        #ax.grid(True)
        if row_indx == 0:
            ax.set_xticklabels([])
        if col_indx > 0:
            ax.set_yticklabels([])
        #ax.text(60,10,"%s" % dirs[i],fontsize=35,bbox=dict(facecolor='white'))
        #ax.text(60,10,"%s" % dirs[i],fontsize=20)
        ax.text(40,88,"%s" % dirs[i],fontsize=20)
        ax.set_autoscale_on(False)

        #eps_for_ryan = "#%5s%5s%10s\n" % ("i","j","epsilon")
        #for n in range(len(contacts)):
        #    eps_for_ryan += "%5d%5d%10.5f\n" % (contacts[n,0],contacts[n,1],eps[n])
        #open("%s_map_%d_Vanilla" % (dirs[i],iteration),"w").write(eps_for_ryan)
        os.chdir("../..")

    if not os.path.exists("plots"):
        os.mkdir("plots")

    #ax0.set_xlabel("Residue i")
    #ax0.set_ylabel("Residue i")
    ax0.text(0.32,1.01,"TS Contact Energy Fluctuations",fontsize=20)
    ax0.text(0.45,-0.06,"Residue i",fontsize=20)
    ax0.text(-0.05,0.55,"Residue j",fontsize=20,rotation="vertical")
    ax0.set_yticks([])
    ax0.set_xticks([])

    fig.subplots_adjust(right=0.88,wspace=0,hspace=0)
    #fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.9, 0.2, 0.025, 0.6])
    cbar = fig.colorbar(image, cax=cbar_ax)
    cbar.set_label("$\\sigma^2_{E_{ij}}$ (fraction of max)",fontsize=20)
    
    #fig.suptitle("TS Energy Fluctuations Iteration %d" % iteration,fontsize=20)
    #fig.suptitle("TS Contact Energy Fluctuations",fontsize=20)
    plt.savefig("plots/S6_TS_fluct_%d.png" % iteration,dpi=800,bbox_inches='tight')
    plt.savefig("plots/S6_TS_fluct_%d.pdf" % iteration,dpi=800,bbox_inches='tight')
    plt.savefig("plots/S6_TS_fluct_%d.eps" % iteration,dpi=800,bbox_inches='tight')


    plt.show()

def plot_contact_phi_maps(iteration):

    #cmap = plt.get_cmap("gnuplot2")
    cmap = cubecmap

    cmap.set_bad(color='w',alpha=1.)
    #cmap.set_bad(color='gray',alpha=1)

    permutants = [13,33,54,68,81]
    dirs = ["S6"] + [ "cp%d" % x for x in permutants ]

    n_residues = len(open("S6/Native.pdb","r").readlines())
    #iteration = 3
    #emin,emax = np.loadtxt("overall_epsilon_range",unpack=True)
    emin = 0
    emax = 1

    fig = plt.figure(figsize=(18,9))
    #fig = plt.figure()
    ax0 = fig.add_subplot(111)
    axes = np.array([[fig.add_subplot(231),fig.add_subplot(232),fig.add_subplot(233)],
                     [fig.add_subplot(234),fig.add_subplot(235),fig.add_subplot(236)]])
    #fig,axes = plt.subplots(nrows=2,ncols=3,figsize=(18,9))

    all_Cs = []
    for i in range(len(dirs)):
        os.chdir("%s/iteration_%d" % (dirs[i],iteration))
        Temp = open("long_temps_last" ,"r").readlines()[0].rstrip("\n")
        contacts = np.loadtxt("%s/native_contacts.ndx" % Temp,skiprows=1)
        
        TS = np.loadtxt("cont_prob_TS.dat")
        U = np.loadtxt("cont_prob_U.dat")
        N = np.loadtxt("cont_prob_N.dat")
        cont_phi = (TS - U)/(N - U)
        #eps_fluct = np.loadtxt("fluct/eps_fluct_TS_U_or_N.dat")
        #eps_fluct = (eps_fluct - eps_fluct.min())/eps_fluct.max()

        col_indx = i % 3
        row_indx = i / 3
        ax = axes[row_indx,col_indx]

        C = np.zeros((n_residues,n_residues),float)*np.nan
        if i == 0:
            for k in range(len(cont_phi)):
                i_idx = contacts[k,1] 
                j_idx = contacts[k,0]
                C[i_idx-1,j_idx-1] = cont_phi[k]
        else:
            p = permutants[i-1]
            for k in range(len(cont_phi)):
                i_idx = contacts[k,1] 
                j_idx = contacts[k,0]
                new_i_idx = revert_index(i_idx,p,n_residues)
                new_j_idx = revert_index(j_idx,p,n_residues)
                if new_j_idx < new_i_idx:
                    C[new_i_idx-1,new_j_idx-1] = cont_phi[k]
                else:
                    C[new_j_idx-1,new_i_idx-1] = cont_phi[k]

        all_Cs.append(C)
        #C = np.ma.masked_invalid(C)
        C = np.ma.array(C,mask=np.isnan(C))
        image = ax.pcolormesh(C,vmin=emin,vmax=emax,cmap=cmap)
        ax.set_xlim(0,n_residues)
        ax.set_ylim(0,n_residues)
        #ax.set_aspect('equal', 'datalim')
        ax.set_xticks(range(0,n_residues,10))
        ax.set_yticks(range(0,n_residues,10))
        #ax.grid(True)
        if row_indx == 0:
            ax.set_xticklabels([])
        if col_indx > 0:
            ax.set_yticklabels([])
        #ax.text(60,10,"%s" % dirs[i],fontsize=35,bbox=dict(facecolor='white'))
        #ax.text(60,10,"%s" % dirs[i],fontsize=20)
        ax.text(40,88,"%s" % dirs[i],fontsize=20)
        ax.set_autoscale_on(False)

        #eps_for_ryan = "#%5s%5s%10s\n" % ("i","j","epsilon")
        #for n in range(len(contacts)):
        #    eps_for_ryan += "%5d%5d%10.5f\n" % (contacts[n,0],contacts[n,1],eps[n])
        #open("%s_map_%d_Vanilla" % (dirs[i],iteration),"w").write(eps_for_ryan)
        os.chdir("../..")

    if not os.path.exists("plots"):
        os.mkdir("plots")

    #ax0.set_xlabel("Residue i")
    #ax0.set_ylabel("Residue i")
    ax0.text(0.4,1.01,"Contact probability $\\phi$",fontsize=20)
    ax0.text(0.45,-0.06,"Residue i",fontsize=20)
    ax0.text(-0.05,0.55,"Residue j",fontsize=20,rotation="vertical")
    ax0.set_yticks([])
    ax0.set_xticks([])

    fig.subplots_adjust(right=0.88,wspace=0,hspace=0)
    #fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.9, 0.2, 0.025, 0.6])
    cbar = fig.colorbar(image, cax=cbar_ax)
    cbar.set_label("Contact $\\phi$ ",fontsize=20)
    
    #fig.suptitle("TS Energy Fluctuations Iteration %d" % iteration,fontsize=20)
    #fig.suptitle("TS Contact Energy Fluctuations",fontsize=20)
    plt.savefig("plots/S6_TS_contact_phi_%d.png" % iteration,dpi=800,bbox_inches='tight')
    plt.savefig("plots/S6_TS_contact_phi_%d.pdf" % iteration,dpi=800,bbox_inches='tight')
    plt.savefig("plots/S6_TS_contact_phi_%d.eps" % iteration,dpi=800,bbox_inches='tight')

    plt.show()

if __name__ == "__main__":
    iteration = 0
    #plot_fluct_maps(iteration)
    plot_contact_phi_maps(iteration)
