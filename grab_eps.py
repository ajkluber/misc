import numpy as np
import matplotlib.pyplot as plt

from contact_phi import *

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

def calc_avg_eps_ss(name,iteration):
    myconts = np.loadtxt("%s/iteration_%d/newton/pairwise_params" % (name,iteration-1),usecols=(0,1),dtype=int)
    myint_type = np.loadtxt("%s/iteration_%d/newton/pairwise_params" % (name,iteration-1),usecols=(3,),dtype=int)
    myeps = np.loadtxt("%s/iteration_%d/newton/model_params" % (name,iteration-1))
    myeps[myint_type == 3] = -1*myeps[myint_type == 3]

    ryanconts = np.loadtxt("../DCA/%s_pairwise_params_0.25" % name,usecols=(0,1),dtype=int)
    ryanint_type = np.loadtxt("../DCA/%s_pairwise_params_0.25" % name,usecols=(3,),dtype=int)
    ryaneps = np.loadtxt("../DCA/%s_model_params_0.25" % name)
    ryaneps[ryanint_type == 3] = -1*ryaneps[ryanint_type == 3]

    element, bounds = get_sec_structure(name)

    mine = []
    ryan = []
    for i in range(len(element)):

        myconts_in_ss = ((myconts[:,0] >= bounds[i,0]).astype(int)*(myconts[:,0] <= bounds[i,1]).astype(int) +\
                        (myconts[:,1] >= bounds[i,0]).astype(int)*(myconts[:,1] <= bounds[i,1]).astype(int)).astype(bool)

        ryanconts_in_ss = ((ryanconts[:,0] >= bounds[i,0]).astype(int)*(ryanconts[:,0] <= bounds[i,1]).astype(int) +\
                        (ryanconts[:,1] >= bounds[i,0]).astype(int)*(ryanconts[:,1] <= bounds[i,1]).astype(int)).astype(bool)

        x = np.mean(myeps[myconts_in_ss])
        y = np.mean(ryaneps[ryanconts_in_ss])
        mine.append(x)
        ryan.append(y)
        #print element[i], x, y

    mine = np.array(mine)    
    ryan = np.array(ryan)
    return element, mine, ryan

def plot_average_eps_over_ss(iteration):
    permutants = [13,33,54,68,81]
    dirs = ["S6"] + [ "cp%d" % x for x in permutants ]


    fig,axes = plt.subplots(nrows=2,ncols=3,figsize=(18,9))
    all_eps = []
    for i in range(len(dirs)):
        name = dirs[i]

        col_indx = i % 3
        row_indx = i / 3
        ax = axes[row_indx,col_indx]

        element, mine, ryan = calc_avg_eps_ss(name,iteration)
        #print min([min(mine),min(ryan)]), max([max(mine),max(ryan)])

        ## Calculate least squares fit and R^2 value.
        x = np.array(mine)
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A,ryan)[0]
        fit = m*x + c*np.ones(len(x))

        SEy = np.sum((ryan - np.mean(ryan))**2)
        SEline = np.sum((ryan - fit)**2)
        r2 = 1. - (SEline/SEy)
        if c < 0.:
            ax.plot(x,fit,'b',lw=2,label="y = %.2fx - %.2f\nR$^2$=%.3f" % (m,abs(c),r2))
        else:
            ax.plot(x,fit,'b',lw=2,label="y = %.2fx + %.2f\nR$^2$=%.3f" % (m,c,r2))
        for n in range(len(mine)):
            ax.plot(mine[n],ryan[n],'ro')

        ax.legend(loc=2,fontsize=15)

        ax.set_xlim(0.85,1.32)
        ax.set_ylim(0.85,1.32)
        ax.grid(True)
        if row_indx == 0:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Alex $\\langle\\epsilon_{ij}\\rangle_{sec struct}$",fontsize=20)
        if col_indx != 0:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("Ryan $\\langle\\epsilon_{ij}\\rangle_{sec struct}$",fontsize=20)
        ax.text(0.86,1.1,"%s" % name,fontsize=30,bbox=dict(facecolor='white'))

    fig.subplots_adjust(wspace=0,hspace=0)
    fig.suptitle("DCA Chi=0.25 vs. Alex contact energy averaged over sec. struct.  %d" % iteration,fontsize=30)
    plt.savefig("plots/DCA_Alex_eps_ss_%d.pdf" % iteration)
    plt.savefig("plots/DCA_Alex_eps_ss_%d.png" % iteration)
    plt.show()

def plot_eps_maps(iteration):
    permutants = [13,33,54,68,81]
    dirs = ["S6"] + [ "cp%d" % x for x in permutants ]

    n_residues = len(open("S6/Native.pdb","r").readlines())
    #iteration = 3
    #emin,emax = np.loadtxt("overall_epsilon_range",unpack=True)
    emin = -0.5
    emax = 2

    fig,axes = plt.subplots(nrows=2,ncols=3,figsize=(18,9))
    all_Cs = []
    for i in range(len(dirs)):
        contacts = np.loadtxt("%s/iteration_%d/newton/pairwise_params" % (dirs[i],iteration-1),usecols=(0,1),dtype=int)
        int_type = np.loadtxt("%s/iteration_%d/newton/pairwise_params" % (dirs[i],iteration-1),usecols=(3,),dtype=int)
        eps = np.loadtxt("%s/iteration_%d/newton/model_params" % (dirs[i],iteration-1))

        eps[int_type == 3] = -1*eps[int_type == 3]


        col_indx = i % 3
        row_indx = i / 3
        ax = axes[row_indx,col_indx]

        C = np.zeros((n_residues,n_residues),float)
        if i == 0:
            for k in range(len(eps)):
                i_idx = contacts[k,1] 
                j_idx = contacts[k,0]
                C[i_idx-1,j_idx-1] = eps[k]
        else:
            p = permutants[i-1]
            for k in range(len(eps)):
                i_idx = contacts[k,1] 
                j_idx = contacts[k,0]
                new_i_idx = revert_index(i_idx,p,n_residues)
                new_j_idx = revert_index(j_idx,p,n_residues)
                if new_j_idx < new_i_idx:
                    C[new_i_idx-1,new_j_idx-1] = eps[k]
                else:
                    C[new_j_idx-1,new_i_idx-1] = eps[k]

        all_Cs.append(C)
        image = ax.pcolor(C,vmin=emin,vmax=emax)
        ax.set_xlim(0,n_residues)
        ax.set_ylim(0,n_residues)
        ax.set_xticks(range(0,n_residues,10))
        ax.set_yticks(range(0,n_residues,10))
        ax.grid(True)
        if row_indx == 0:
            ax.set_xticklabels([])
        if col_indx == 1:
            ax.set_yticklabels([])
        ax.text(60,10,"%s" % dirs[i],fontsize=35,bbox=dict(facecolor='white'))

        #eps_for_ryan = "#%5s%5s%10s\n" % ("i","j","epsilon")
        #for n in range(len(contacts)):
        #    eps_for_ryan += "%5d%5d%10.5f\n" % (contacts[n,0],contacts[n,1],eps[n])
        #open("%s_map_%d_Vanilla" % (dirs[i],iteration),"w").write(eps_for_ryan)

    fig.subplots_adjust(right=0.88,wspace=0,hspace=0)
    #fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.9, 0.2, 0.025, 0.6])
    fig.colorbar(image, cax=cbar_ax)
    fig.suptitle("Contact Epsilons Iteration %d" % iteration,fontsize=30)
    #plt.savefig("plots/S6_epsilons_%d.pdf" % iteration)
    #plt.savefig("plots/S6_epsilons_%d.png" % iteration)
    #plt.show()

    all_Cs = np.array(all_Cs)
    avgC = np.mean(all_Cs,axis=0)
    stdC = np.std(all_Cs,axis=0)
    plt.figure()
    plt.pcolor(avgC,vmin=emin,vmax=emax)
    plt.colorbar()
    plt.title("S6 average contact energies over permutants",fontsize=17)
    plt.xlim(0,n_residues)
    plt.ylim(0,n_residues)
    plt.xticks(range(0,n_residues,10))
    plt.yticks(range(0,n_residues,10))
    plt.savefig("plots/S6_avg_eps_%d.pdf" % iteration)
    plt.savefig("plots/S6_avg_eps_%d.png" % iteration)
    plt.grid(True)

    plt.figure()
    plt.pcolor(stdC)
    plt.colorbar()
    plt.xlim(0,n_residues)
    plt.ylim(0,n_residues)
    plt.xticks(range(0,n_residues,10))
    plt.yticks(range(0,n_residues,10))
    plt.grid(True)
    plt.title("S6 std. dev. contact energies over permutants",fontsize=17)
    plt.savefig("plots/S6_std_eps_%d.pdf" % iteration)
    plt.savefig("plots/S6_std_eps_%d.png" % iteration)
    plt.show()


def calc_eps_correlation(iteration):
    iteration = 4
    permutants = [13,33,54,68,81]
    dirs = ["S6"] + [ "cp%d" % x for x in permutants ]

    emin = -2
    emax = 5

    n_residues = len(open("S6/Native.pdb","r").readlines())
    n_prots = len(dirs)
    
    fig,axes = plt.subplots(nrows=n_prots,ncols=n_prots,figsize=(20,20))

    for i in range(len(dirs)):
        name_A = dirs[i]
        eps_A = np.loadtxt("%s/iteration_%d/newton/model_params" % (name_A,iteration-1))
        type_A = np.loadtxt("%s/iteration_%d/newton/pairwise_params" % (name_A,iteration-1),usecols=(3,),dtype=int)
        eps_A[type_A == 3] = -1*eps_A[type_A == 3]
        n_eps = float(len(eps_A))

        for j in range(len(dirs)):
            name_B = dirs[j]
            eps_B = np.loadtxt("%s/iteration_%d/newton/model_params" % (name_B,iteration-1))
            type_B = np.loadtxt("%s/iteration_%d/newton/pairwise_params" % (name_B,iteration-1),usecols=(3,),dtype=int)
            eps_B[type_B == 3] = -1*eps_B[type_B == 3]

            corrAB = np.mean((eps_A - np.mean(eps_A)*np.ones(len(eps_A)))*(eps_B - np.mean(eps_B)*np.ones(len(eps_B))))/(np.std(eps_A)*np.std(eps_B))
            ax = axes[i,j]

            ## Calculate least squares fit and R^2 value.
            x = np.array(eps_A)
            A = np.vstack([x, np.ones(len(x))]).T
            m, c = np.linalg.lstsq(A,eps_B)[0]
            fit = m*x + c*np.ones(len(x))

            SEy = np.sum((eps_B - np.mean(eps_B))**2)
            SEline = np.sum((eps_B - fit)**2)
            r2 = 1. - (SEline/SEy)
            ax.set_xlim(emin,emax)
            ax.set_ylim(emin,emax)
            ax.grid(True)
            if i != (len(dirs)-1):
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("%s" % name_B,fontsize=20)
            if j != 0:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel("%s" % name_A,fontsize=20)

            if i > j:
                if c < 0.:
                    ax.plot(x,fit,'b',lw=2,label="y = %.2fx - %.2f\nR$^2$=%.3f\nC=%.3f" % (m,abs(c),r2,corrAB))
                else:
                    ax.plot(x,fit,'b',lw=2,label="y = %.2fx + %.2f\nR$^2$=%.3f\nCorr=%.3f" % (m,c,r2,corrAB))

                ax.plot(eps_A,eps_B,'ro')
                lg = ax.legend(loc=2,fontsize=10)


    fig.subplots_adjust(wspace=0,hspace=0)
    fig.suptitle("Correlation between permutant epsilons Iteration %d" % iteration,fontsize=30)
    plt.savefig("plots/S6_correlation_epsilons_%d.pdf" % iteration)
    plt.savefig("plots/S6_correlation_epsilons_%d.png" % iteration)
    plt.show()

if __name__ == "__main__":
    iteration = 3
    #plot_eps_maps(iteration)
    #plot_average_eps_over_ss(iteration)
    calc_eps_correlation(iteration)
