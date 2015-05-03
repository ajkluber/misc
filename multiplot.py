import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from contact_phi import *

global NAMES
#NAMES = ["S6","cp13","cp33","cp54","cp68","cp81"]
#NAMES = ["S6","cp68","cp81"]
NAMES = ["cp13","cp33","cp54"]
#NAMES = ["S6","cp13","cp33"]
#NAMES = ["cp54","cp68","cp81"]
global ITERATIONS
ITERATIONS = [0,2]

def get_sec_struct_groups(name):
    """ Get a user defined set of Qfoldons from file """
    if not os.path.exists("%s/Qgroups_ss" % name):
        print "ERROR! %s/Qfoldons DOES NOT EXIST!" % name
        print " Specify sets of contacts"
        print "Exiting"
        raise SystemExit

    labels = [ x.rstrip("\n") for x in open("%s/Qgroups_ss/labels.txt" % name,"r").readlines() ]
    colors = [ x.rstrip("\n") for x in open("%s/Qgroups_ss/colors.txt" % name,"r").readlines() ]
    n_grps = len(colors)

    Qgrp_indxs = []
    for i in range(n_grps):
        Qgrp_indxs.append(np.loadtxt("%s/Qgroups_ss/group%d.dat" % (name,i),dtype=int))

    return Qgrp_indxs, n_grps, colors, labels

def get_foldon_groups(name):
    """ Get a user defined set of Qfoldons from file """
    if not os.path.exists("%s/Qfoldons" % name):
        print "ERROR! %s/Qfoldons DOES NOT EXIST!" % name
        print " Specify sets of contacts"
        print "Exiting"
        raise SystemExit

    labels = [ x.rstrip("\n") for x in open("%s/Qfoldons/labels.txt" % name,"r").readlines() ]
    colors = [ x.rstrip("\n") for x in open("%s/Qfoldons/colors.txt" % name,"r").readlines() ]
    n_grps = len(colors)

    Qgrp_indxs = []
    for i in range(n_grps):
        Qgrp_indxs.append(np.loadtxt("%s/Qfoldons/group%d.dat" % (name,i),dtype=int))

    return Qgrp_indxs, n_grps, colors, labels

def plot_all_contact_phi_res():
    """ Plot contact probability phi-values by residue """ 
    
    fig, axes = plt.subplots(nrows=6,ncols=2,figsize=(9,17))

    all_phi_res = []
    for i in range(len(NAMES)):
        temp_phi = []
        for j in range(len(ITERATIONS)):
            ax = axes[i,j]
            exp_phi,mut_indxs = get_exp_phi_res(NAMES[i]) 
            sim_phi1 = calculate_contact_phi_res(NAMES[i],ITERATIONS[j])
            sim_phi = sim_phi1[mut_indxs - 1]

            ## Calculate least squares fit and R^2 value.
            x = np.array(sim_phi)
            A = np.vstack([x, np.ones(len(x))]).T
            m, c = np.linalg.lstsq(A,exp_phi)[0]
            fit = m*x + c*np.ones(len(x))

            SEy = np.sum((exp_phi - np.mean(exp_phi))**2)
            SEline = np.sum((exp_phi - fit)**2)
            r2 = 1. - (SEline/SEy)
            ax.plot([0,1],[0,1],'k',lw=2)
            if c < 0.:
                ax.plot(x,fit,'b',lw=2,label="y = %.2fx - %.2f\nR$^2$=%.3f" % (m,abs(c),r2))
            else:
                ax.plot(x,fit,'b',lw=2,label="y = %.2fx + %.2f\nR$^2$=%.3f" % (m,c,r2))
            for n in range(len(sim_phi)):
                ax.plot(sim_phi[n],exp_phi[n],'ro')
            lg = ax.legend(loc=2,fontsize=10)
            lg.draw_frame(False)
            ax.grid(True)
            ax.set_xlim(0,1) 
            ax.set_ylim(0,1) 
            if i == 0:
                if j == 0:
                    ax.set_title("Vanilla",fontsize=18)
                else:
                    ax.set_title("Het. Iter. %d" % ITERATIONS[j],fontsize=18)
            if j == 1: 
                ax.set_aspect("equal",anchor="SW")
                ax.set_yticklabels([]) 
                if i != (len(NAMES)-1):
                    ax.set_yticklabels([]) 
                else:
                    ax.set_xlabel("$\phi_{sim}$", fontsize=28)
            else:
                ax.set_aspect("equal",anchor="SE")
                ax.set_ylabel("%s\n$\phi_{exp}$" % NAMES[i],fontsize=22) 
                if i == (len(NAMES)-1):
                    ax.set_xlabel("$\phi_{sim}$", fontsize=28)

    fig.subplots_adjust(hspace=0,wspace=0)  
    if not os.path.exists("plots"):
        os.mkdir("plots")
    fig.suptitle("$\phi_{sim} = \\frac{Q_{TS} - Q_U}{Q_N - Q_U}}$ by Contact probability",fontsize=30)
    plt.savefig("plots/S6_all_contact_phi.pdf")
    plt.savefig("plots/S6_all_contact_phi.png")
    plt.show()

def plot_all_contact_phi_ss():
    """ Plot contact probability phi-values averaged over secondary structure. """ 
    
    fig, axes = plt.subplots(nrows=6,ncols=2,figsize=(9,17))

    all_phi_res = []
    for i in range(len(NAMES)):
        temp_phi = []
        for j in range(len(ITERATIONS)):
            ax = axes[i,j]

            sim_phi = calculate_average_contact_phi_ss(NAMES[i],ITERATIONS[j])
            exp_phi,element = get_exp_phi_ss(NAMES[i])

            ## Calculate least squares fit and R^2 value.
            x = np.array(sim_phi)
            A = np.vstack([x, np.ones(len(x))]).T
            m, c = np.linalg.lstsq(A,exp_phi)[0]
            fit = m*x + c*np.ones(len(x))

            SEy = np.sum((exp_phi - np.mean(exp_phi))**2)
            SEline = np.sum((exp_phi - fit)**2)
            r2 = 1. - (SEline/SEy)
            ax.plot([0,1],[0,1],'k',lw=2)
            if c < 0.:
                ax.plot(x,fit,'b',lw=2,label="y = %.2fx - %.2f\nR$^2$=%.3f" % (m,abs(c),r2))
            else:
                ax.plot(x,fit,'b',lw=2,label="y = %.2fx + %.2f\nR$^2$=%.3f" % (m,c,r2))
            for n in range(len(sim_phi)):
                ax.plot(sim_phi[n],exp_phi[n],'ro')
            lg = ax.legend(loc=2,fontsize=10)
            lg.draw_frame(False)
            ax.grid(True)
            ax.set_xlim(0,1) 
            ax.set_ylim(0,1) 
            if i == 0:
                if j == 0:
                    ax.set_title("Vanilla",fontsize=18)
                else:
                    ax.set_title("Het. Iter. %d" % ITERATIONS[j],fontsize=18)
            if j == 1: 
                ax.set_aspect("equal",anchor="SW")
                ax.set_yticklabels([]) 
                if i != (len(NAMES)-1):
                    ax.set_yticklabels([]) 
                else:
                    ax.set_xlabel("$\phi^{ss}_{sim}$", fontsize=28)
            else:
                ax.set_aspect("equal",anchor="SE")
                ax.set_ylabel("%s\n$\phi^{ss}_{exp}$" % NAMES[i],fontsize=22) 
                if i == (len(NAMES)-1):
                    ax.set_xlabel("$\phi^{ss}_{sim}$", fontsize=28)

    fig.subplots_adjust(hspace=0,wspace=0)        ## plot data
    if not os.path.exists("plots"):
        os.mkdir("plots")
    fig.suptitle("$\phi^{ss}_{sim} = \\langle\\frac{Q_{TS} - Q_U}{Q_N - Q_U}\\rangle_{ss}$ by Contact probability",fontsize=30)
    plt.savefig("plots/S6_all_contact_phi_ss.pdf")
    plt.savefig("plots/S6_all_contact_phi_ss.png")
    plt.show()

def plot_all_ddG_phi_res():
    """ Plot ddG phi-values by residue """ 
    fig, axes = plt.subplots(nrows=6,ncols=2,figsize=(9,17))
    
    for i in range(len(NAMES)):
        temp_phi = []
        for j in range(len(ITERATIONS)):
            ax = axes[i,j]
            exp_phi,mut_indxs = get_exp_phi_res(NAMES[i]) 
            sim_phi = get_sim_ddG_phi_res(NAMES[i],ITERATIONS[j])
            #temp_phi.append([sim_phi,exp_phi])

            ## Calculate least squares fit and R^2 value.
            x = np.array(sim_phi)
            A = np.vstack([x, np.ones(len(x))]).T
            m, c = np.linalg.lstsq(A,exp_phi)[0]
            fit = m*x + c*np.ones(len(x))

            SEy = np.sum((exp_phi - np.mean(exp_phi))**2)
            SEline = np.sum((exp_phi - fit)**2)
            r2 = 1. - (SEline/SEy)
            ax.plot([0,1],[0,1],'k',lw=2)
            if c < 0.:
                ax.plot(x,fit,'b',lw=2,label="y = %.2fx - %.2f\nR$^2$=%.3f" % (m,abs(c),r2))
            else:
                ax.plot(x,fit,'b',lw=2,label="y = %.2fx + %.2f\nR$^2$=%.3f" % (m,c,r2))
            for n in range(len(sim_phi)):
                ax.plot(sim_phi[n],exp_phi[n],'ro')
            lg = ax.legend(loc=2,fontsize=10)
            lg.draw_frame(False)
            ax.grid(True)
            ax.set_xlim(0,1) 
            ax.set_ylim(0,1) 
            if i == 0:
                if j == 0:
                    ax.set_title("Vanilla",fontsize=18)
                else:
                    ax.set_title("Het. Iter. %d" % ITERATIONS[j],fontsize=18)
            if j == 1: 
                ax.set_aspect("equal",anchor="SW")
                ax.set_yticklabels([]) 
                if i != (len(NAMES)-1):
                    ax.set_yticklabels([]) 
                else:
                    ax.set_xlabel("$\phi_{sim}$", fontsize=28)
            else:
                ax.set_aspect("equal",anchor="SE")
                ax.set_ylabel("%s\n$\phi_{exp}$" % NAMES[i],fontsize=22) 
                if i == (len(NAMES)-1):
                    ax.set_xlabel("$\phi_{sim}$", fontsize=28)

    fig.subplots_adjust(hspace=0,wspace=0)        ## plot data
    if not os.path.exists("plots"):
        os.mkdir("plots")
    fig.suptitle("$\phi_{sim} = \\frac{\\Delta\\Delta G^{TS}}{\\Delta\\Delta G^{N}}$ by Free Energy Perturbation",fontsize=30)
    plt.savefig("plots/S6_all_ddG_phi.pdf")
    plt.savefig("plots/S6_all_ddG_phi.png")
    plt.show()

def plot_all_ddG_phi_ss():
    """ Plot ddG phi-values averaged over secondary structure. """ 
    
    fig, axes = plt.subplots(nrows=6,ncols=2,figsize=(9,17))

    all_phi_res = []
    for i in range(len(NAMES)):
        temp_phi = []
        for j in range(len(ITERATIONS)):
            ax = axes[i,j]

            sim_phi = get_sim_ddG_phi_ss(NAMES[i],ITERATIONS[j])
            exp_phi,element = get_exp_phi_ss(NAMES[i])

            #print sim_phi
            #raise SystemExit

            ## Calculate least squares fit and R^2 value.
            x = np.array(sim_phi)
            A = np.vstack([x, np.ones(len(x))]).T
            m, c = np.linalg.lstsq(A,exp_phi)[0]
            fit = m*x + c*np.ones(len(x))

            SEy = np.sum((exp_phi - np.mean(exp_phi))**2)
            SEline = np.sum((exp_phi - fit)**2)
            r2 = 1. - (SEline/SEy)
            ax.plot([0,1],[0,1],'k',lw=2)
            if c < 0.:
                ax.plot(x,fit,'b',lw=2,label="y = %.2fx - %.2f\nR$^2$=%.3f" % (m,abs(c),r2))
            else:
                ax.plot(x,fit,'b',lw=2,label="y = %.2fx + %.2f\nR$^2$=%.3f" % (m,c,r2))
            for n in range(len(sim_phi)):
                ax.plot(sim_phi[n],exp_phi[n],'ro')
            lg = ax.legend(loc=2,fontsize=10)
            lg.draw_frame(False)
            ax.grid(True)
            ax.set_xlim(0,1) 
            ax.set_ylim(0,1) 
            if i == 0:
                if j == 0:
                    ax.set_title("Vanilla",fontsize=18)
                else:
                    ax.set_title("Het. Iter. %d" % ITERATIONS[j],fontsize=18)
            if j == 1: 
                ax.set_aspect("equal",anchor="SW")
                ax.set_yticklabels([]) 
                if i != (len(NAMES)-1):
                    ax.set_yticklabels([]) 
                else:
                    ax.set_xlabel("$\phi^{ss}_{sim}$", fontsize=28)
            else:
                ax.set_aspect("equal",anchor="SE")
                ax.set_ylabel("%s\n$\phi^{ss}_{exp}$" % NAMES[i],fontsize=22) 
                if i == (len(NAMES)-1):
                    ax.set_xlabel("$\phi^{ss}_{sim}$", fontsize=28)

    fig.subplots_adjust(hspace=0,wspace=0)        ## plot data
    if not os.path.exists("plots"):
        os.mkdir("plots")
    fig.suptitle("$\phi^{ss}_{sim} = \\langle\\frac{\\Delta\\Delta G^{TS}}{\\Delta\\Delta G^{N}}\\rangle_{ss}$ by Free Energy Perturbation",fontsize=30)
    plt.savefig("plots/S6_all_ddG_phi_ss.pdf")
    plt.savefig("plots/S6_all_ddG_phi_ss.png")
    plt.show()

def plot_all_ddG():
    """ Plot ddG simulation vs experiment """ 
    
    fig, axes = plt.subplots(nrows=6,ncols=2,figsize=(9,17))

    for i in range(len(NAMES)):
        temp_phi = []
        for j in range(len(ITERATIONS)):
            ax = axes[i,j]

            sim = np.loadtxt("%s/iteration_%d/newton/sim_feature.dat" % (NAMES[i],ITERATIONS[j]))
            exp = np.loadtxt("%s/iteration_%d/newton/target_feature.dat" % (NAMES[i],ITERATIONS[j]))
            N = len(sim)/2
            maxddg = max(exp)+0.5
            minddg = min(exp)-0.5

            ## Calculate least squares fit and R^2 value.
            x = np.array(sim)
            A = np.vstack([x, np.ones(len(x))]).T
            m, c = np.linalg.lstsq(A,exp)[0]
            fit = m*x + c*np.ones(len(x))

            SEy = np.sum((exp - np.mean(exp))**2)
            SEline = np.sum((exp - fit)**2)
            r2 = 1. - (SEline/SEy)
            ax.plot([minddg,maxddg],[minddg,maxddg],'k',lw=2)
            if c < 0.:
                ax.plot(x,fit,'b',lw=2,label="y = %.2fx - %.2f\nR$^2$=%.3f" % (m,abs(c),r2))
            else:
                ax.plot(x,fit,'b',lw=2,label="y = %.2fx + %.2f\nR$^2$=%.3f" % (m,c,r2))

            for n in range(N):
                ax.plot(sim[n],exp[n],'bo')
            for n in range(N):
                ax.plot(sim[n+N],exp[n+N],'ro')

            lg = ax.legend(loc=4,fontsize=8)
            #lg.draw_frame(False)
            ax.grid(True)
            ax.set_xlim(minddg,maxddg) 
            ax.set_ylim(minddg,maxddg) 
            if i == 0:
                if j == 0:
                    ax.set_title("Vanilla",fontsize=18)
                else:
                    ax.set_title("Het. Iter. %d" % ITERATIONS[j],fontsize=18)
            if j == 1: 
                ax.set_aspect("equal",anchor="SW")
                ax.set_yticklabels([]) 
                if i != (len(NAMES)-1):
                    ax.set_yticklabels([]) 
                else:
                    ax.set_xlabel("$\Delta\Delta G_{sim}$ (k$_B$T$_{f}$)", fontsize=18)
            else:
                ax.set_aspect("equal",anchor="SE")
                ax.set_ylabel("%s\n$\Delta\Delta G_{exp}$ (k$_B$ T$_{exp}$)" % NAMES[i],fontsize=16) 
                if i == (len(NAMES)-1):
                    ax.set_xlabel("$\Delta\Delta G_{sim}$ (k$_B$T$_{f}$)", fontsize=18)

    fig.subplots_adjust(hspace=0,wspace=0)        ## plot data
    if not os.path.exists("plots"):
        os.mkdir("plots")
    fig.suptitle("Comparing $\\Delta\\Delta G$'s",fontsize=30)
    plt.savefig("plots/S6_all_ddG.pdf")
    plt.savefig("plots/S6_all_ddG.png")
    plt.show()

def plot_all_Qss_vs_Q():
    """ Folding mechanism by contact formation versus Q """

    fig, axes = plt.subplots(nrows=6,ncols=2,figsize=(10,17))
    
    all_phi_res = []
    for i in range(len(NAMES)):
        temp_phi = []
        Qgrp_indxs, n_grps, colors, labels = get_sec_struct_groups(NAMES[i])
        n_contacts = len(open("%s/contacts.dat" % NAMES[i],"r").readlines())
        for j in range(len(ITERATIONS)):
            ax = axes[i,j]

            Qi_vs_Q = np.loadtxt("%s/iteration_%d/QivsQ.dat" % (NAMES[i],ITERATIONS[j]))
            Qbins = np.loadtxt("%s/iteration_%d/Qbins.dat" % (NAMES[i],ITERATIONS[j]))
            n_bins = len(Qbins)

            Qgrp_vs_Q = np.zeros((n_bins,n_grps),float)
            for n in range(n_grps):
                if Qgrp_indxs[n].shape == ():
                    Qgrp_vs_Q[:,n] = Qi_vs_Q[:,Qgrp_indxs[n]]
                else: 
                    n_grp_members = Qgrp_indxs[n].shape[0]
                    Qgrp_vs_Q[:,n] = sum(Qi_vs_Q[:,Qgrp_indxs[n]].T)/n_grp_members
                ax.plot(Qbins,Qgrp_vs_Q[:,n],color=colors[n],lw=2)

            ax.grid(True)
            ax.set_xlim(0,n_contacts) 
            ax.set_ylim(0,1) 
            if i == 0:
                if j == 0:
                    ax.set_title("Vanilla",fontsize=18)
                else:
                    ax.set_title("Het. Iter. %d" % ITERATIONS[j],fontsize=18)
            if j == 1: 
                #ax.set_aspect("equal",anchor="SW")
                ax.set_yticklabels([]) 
                if i != (len(NAMES)-1):
                    ax.set_yticklabels([]) 
                else:
                    ax.set_xlabel("$Q$", fontsize=28)
            else:
                #ax.set_aspect("equal",anchor="SE")
                ax.set_ylabel("%s\n$Q_{ss}$" % NAMES[i],fontsize=22) 
                if i == (len(NAMES)-1):
                    ax.set_xlabel("$Q$", fontsize=28)

    fig.subplots_adjust(hspace=0,wspace=0)        ## plot data
    if not os.path.exists("plots"):
        os.mkdir("plots")
    #fig.suptitle("$\phi_{sim} = \\frac{\\Delta\\Delta G^{TS}}{\\Delta\\Delta G^{N}}$ by Free Energy Perturbation",fontsize=30)
    fig.suptitle("$Q_{ss}$ vs. $Q$ Folding Mechanism",fontsize=30)
    plt.savefig("plots/S6_all_Qss.pdf")
    plt.savefig("plots/S6_all_Qss.png")
    plt.show()

def plot_all_Qfoldon_vs_Q():
    """ Folding mechanism by contact formation versus Q """

    #fig, axes = plt.subplots(nrows=6,ncols=2,figsize=(10,17))
    fig, axes = plt.subplots(nrows=3,ncols=2,figsize=(10,10))
    
    all_phi_res = []
    for i in range(len(NAMES)):
        temp_phi = []
        Qgrp_indxs, n_grps, colors, labels = get_foldon_groups(NAMES[i])
        #n_contacts = len(open("%s/contacts.dat" % NAMES[i],"r").readlines())
        n_contacts = 260
        for j in range(len(ITERATIONS)):
            ax = axes[i,j]

            Qi_vs_Q = np.loadtxt("%s/iteration_%d/QivsQ.dat" % (NAMES[i],ITERATIONS[j]))
            Qbins = np.loadtxt("%s/iteration_%d/Qbins.dat" % (NAMES[i],ITERATIONS[j]))
            n_bins = len(Qbins)

            Qgrp_vs_Q = np.zeros((n_bins,n_grps),float)
            for n in range(n_grps):
                if Qgrp_indxs[n].shape == ():
                    Qgrp_vs_Q[:,n] = Qi_vs_Q[:,Qgrp_indxs[n]]
                else: 
                    n_grp_members = Qgrp_indxs[n].shape[0]
                    Qgrp_vs_Q[:,n] = sum(Qi_vs_Q[:,Qgrp_indxs[n]].T)/n_grp_members
                ax.plot(Qbins,Qgrp_vs_Q[:,n],color=colors[n],lw=2,label=labels[n])

            ax.grid(True)
            ax.set_xlim(0,n_contacts) 
            ax.set_ylim(0,1) 
            if i == 0:
                if j == 0:
                    #ax.set_title("Vanilla",fontsize=18)
                    ax.set_title("Homogeneous",fontsize=20)
                    lg = ax.legend(loc=2,fontsize=10)
                    #lg.draw_frame(False)
                else:
                    #ax.set_title("Het. Iter. %d" % ITERATIONS[j],fontsize=18)
                    ax.set_title("Heterogeneous",fontsize=20)
            if j == 1: 
                #ax.set_aspect("equal",anchor="SW")
                ax.set_yticklabels([]) 
                if i != (len(NAMES)-1):
                    ax.set_yticklabels([]) 
                else:
                    ax.set_xlabel("$Q$", fontsize=28)
            else:
                #ax.set_aspect("equal",anchor="SE")
                ax.set_ylabel("%s\n$Q_{group}$" % NAMES[i],fontsize=22) 
                if i == (len(NAMES)-1):
                    ax.set_xlabel("$Q$", fontsize=28)

    fig.subplots_adjust(hspace=0,wspace=0)        ## plot data
    if not os.path.exists("plots"):
        os.mkdir("plots")
    #fig.suptitle("$\phi_{sim} = \\frac{\\Delta\\Delta G^{TS}}{\\Delta\\Delta G^{N}}$ by Free Energy Perturbation",fontsize=30)
    #fig.suptitle("$Q_{group}$ vs. $Q$ Folding Mechanism",fontsize=30)
    #plt.savefig("plots/S6_cp13_cp33_Qfoldons.pdf")
    #plt.savefig("plots/S6_cp13_cp33_Qfoldons.png")
    #plt.savefig("plots/cp54_cp68_cp81_Qfoldons.pdf")
    #plt.savefig("plots/cp54_cp68_cp81_Qfoldons.png")
    #plt.savefig("plots/S6_cp68_cp81_Qfoldons.pdf")
    #plt.savefig("plots/S6_cp68_cp81_Qfoldons.png")
    plt.savefig("plots/cp13_cp33_cp54_Qfoldons.pdf")
    plt.savefig("plots/cp13_cp33_cp54_Qfoldons.png")
    #plt.savefig("plots/S6_all_Qfoldons.pdf")
    #plt.savefig("plots/S6_all_Qfoldons.png")
    plt.show()

def plot_all_Route_vs_Q():
    """ Folding mechanism by contact formation versus Q """

    fig, axes = plt.subplots(nrows=6,ncols=2,figsize=(10,17))
    
    all_phi_res = []
    for i in range(len(NAMES)):
        temp_phi = []
        #n_contacts = len(open("%s/contacts.dat" % NAMES[i],"r").readlines())
        n_contacts = 260
        for j in range(len(ITERATIONS)):
            ax = axes[i,j]

            Qi_vs_Q = np.loadtxt("%s/iteration_%d/QivsQ.dat" % (NAMES[i],ITERATIONS[j]))
            Qbins = np.loadtxt("%s/iteration_%d/Qbins.dat" % (NAMES[i],ITERATIONS[j]))
            n_bins = len(Qbins)

            Q = Qbins/float(max(Qbins))
            route = np.zeros(n_bins)
            for n in range(n_bins):
                if (Q[n] == 0) or (Q[n] == 1):
                    pass
                else:
                    route[n] = (1./(Q[n]*(1. - Q[n])))*(np.std(Qi_vs_Q[n,:])**2)


            ax.plot(Qbins,route,'r')
            ax.grid(True)
            ax.set_xlim(0,n_contacts) 
            ax.set_ylim(0,0.6) 
            if i == 0:
                if j == 0:
                    ax.set_title("Vanilla",fontsize=18)
                    lg = ax.legend(loc=2,fontsize=10)
                    #lg.draw_frame(False)
                else:
                    ax.set_title("Het. Iter. %d" % ITERATIONS[j],fontsize=18)
            if j == 1: 
                #ax.set_aspect("equal",anchor="SW")
                ax.set_yticklabels([]) 
                if i != (len(NAMES)-1):
                    ax.set_yticklabels([]) 
                else:
                    ax.set_xlabel("$Q$", fontsize=28)
            else:
                #ax.set_aspect("equal",anchor="SE")
                ax.set_ylabel("%s\n$R(Q)$" % NAMES[i],fontsize=22) 
                if i == (len(NAMES)-1):
                    ax.set_xlabel("$Q$", fontsize=28)

    fig.subplots_adjust(hspace=0,wspace=0)        ## plot data
    if not os.path.exists("plots"):
        os.mkdir("plots")
    fig.suptitle("$R(Q)$ vs. $Q$ Folding Mechanism",fontsize=30)
    plt.savefig("plots/S6_route.pdf")
    plt.savefig("plots/S6_route.png")
    plt.show()

def plot_all_F_vs_Q():
    """ Folding mechanism by contact formation versus Q """

    fig, axes = plt.subplots(nrows=6,ncols=2,figsize=(10,17))
    
    all_phi_res = []
    for i in range(len(NAMES)):
        temp_phi = []
        n_contacts = len(open("%s/contacts.dat" % NAMES[i],"r").readlines())
        for j in range(len(ITERATIONS)):
            ax = axes[i,j]

            Temps = [ x.rstrip("\n") for x in open("%s/iteration_%d/long_temps_last" % (NAMES[i],ITERATIONS[j]),"r").readlines() ]

            Fs = []
            F_errs = []
            Bin_centers = []
            for i in range(len(Temps)): 
                filename = "%s/iteration_%d/%s/Q.dat" % (Temps[i],NAMES[i],ITERATIONS[j])
                Fs.append(F)
                F_errs.append(F_err)
                Bin_centers.append(bin_centers)
            Fs = np.array(Fs)
            F_errs = np.array(F_errs)


            ax.plot(Qbins,route,'r')
            ax.grid(True)
            ax.set_xlim(0,n_contacts) 
            ax.set_ylim(0,0.6) 
            if i == 0:
                if j == 0:
                    ax.set_title("Vanilla",fontsize=18)
                    lg = ax.legend(loc=2,fontsize=10)
                    #lg.draw_frame(False)
                else:
                    ax.set_title("Het. Iter. %d" % ITERATIONS[j],fontsize=18)
            if j == 1: 
                #ax.set_aspect("equal",anchor="SW")
                ax.set_yticklabels([]) 
                if i != (len(NAMES)-1):
                    ax.set_yticklabels([]) 
                else:
                    ax.set_xlabel("$Q$", fontsize=28)
            else:
                #ax.set_aspect("equal",anchor="SE")
                ax.set_ylabel("%s\n$R(Q)$" % NAMES[i],fontsize=22) 
                if i == (len(NAMES)-1):
                    ax.set_xlabel("$Q$", fontsize=28)

    fig.subplots_adjust(hspace=0,wspace=0)        ## plot data
    if not os.path.exists("plots"):
        os.mkdir("plots")
    fig.suptitle("$F(Q)$ vs. $Q$ Folding Mechanism",fontsize=30)
    #plt.savefig("plots/S6_FvsQ.pdf")
    #plt.savefig("plots/S6_FvsQ.png")
    plt.show()

if __name__ == "__main__":
    pass
    #plot_all_ddG_phi_res()
    #plot_all_contact_phi_res()
    plot_all_Qfoldon_vs_Q()
    #plot_all_Qss_vs_Q()
    #plot_all_ddG()
    #plot_all_contact_phi_ss()
    #plot_all_ddG_phi_ss()
    #plot_all_Route_vs_Q()
