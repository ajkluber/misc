import argparse
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from misc.cube_cmap import cubecmap, cubeYFcmap

def get_qearly_qlate(name,iteration,nbins=30):
    early = np.loadtxt("%s/iteration_0/early_conts" % name,dtype=int)
    late = np.loadtxt("%s/iteration_0/late_conts" % name,dtype=int)

    os.chdir("%s/iteration_0" % name)
    temps = [ x.rstrip("\n") for x in open("long_temps_last","r").readlines() ]
    qearly0 = np.concatenate([ np.loadtxt("%s/qearly.dat" % x) for x in temps ])
    qlate0 = np.concatenate([ np.loadtxt("%s/qlate.dat" % x) for x in temps ])
    os.chdir("../../%s/iteration_%d" % (name,iteration))
    temps = [ x.rstrip("\n") for x in open("long_temps_last","r").readlines() ]
    qearly_it = np.concatenate([ np.loadtxt("%s/qearly.dat" % x) for x in temps ])
    qlate_it = np.concatenate([ np.loadtxt("%s/qlate.dat" % x) for x in temps ])
    os.chdir("../..")

    n_contacts = len(early) + len(late)
    xbins = np.linspace(0,len(early),nbins)
    ybins = np.linspace(0,len(late),nbins)

    # Get pmf for iteration 0
    #Hxy0, xedges0, yedges0 = np.histogram2d(qearly0,qlate0,bins=nbins)
    Hxy0, xedges0, yedges0 = np.histogram2d(qearly0,qlate0,bins=(xbins,ybins))
    #xcenters0 = 0.5*(xedges0[1:] + xedges0[:-1])
    #ycenters0 = 0.5*(yedges0[1:] + yedges0[:-1])
    xcenters0 = xedges0[:-1]
    ycenters0 = yedges0[:-1]
    X0,Y0 = np.meshgrid(xcenters0,ycenters0)

    pmf0 = -np.log(Hxy0)
    pmf0[pmf0 != np.NaN] -= pmf0[pmf0 != np.NaN].min()

    maskpmf0 = np.ma.array(pmf0,mask=np.isnan(pmf0))

    # Get pmf for iteration 
    Hxy_it, xedges_it, yedges_it = np.histogram2d(qearly_it,qlate_it,bins=(xbins,ybins))
    #xcenters_it = 0.5*(xedges_it[1:] + xedges_it[:-1])
    #ycenters_it = 0.5*(yedges_it[1:] + yedges_it[:-1])
    xcenters_it = xedges_it[:-1]
    ycenters_it = yedges_it[:-1]
    X_it,Y_it = np.meshgrid(xcenters_it,ycenters_it)

    pmf_it = -np.log(Hxy_it)
    pmf_it[pmf_it != np.NaN] -= pmf_it[pmf_it != np.NaN].min()

    maskpmf_it = np.ma.array(pmf_it,mask=np.isnan(pmf_it))
    return X0,Y0,maskpmf0,X_it,Y_it,maskpmf_it 

def plot_Qearly_vs_Qlate(name,iteration,nbins=30):

    # plot
    fig, axes = plt.subplots(1,2,figsize=(8,5.5),sharey=true)

    #cmap = plt.get_cmap("hot")
    #cmap = plt.get_cmap("cool")
    cmap = plt.get_cmap("summer")

    axes[0].set_aspect(1)
    axes[0].contourf(x0,y0,maskpmf0.t,levels=np.arange(0,10,1),cmap=cmap)
    axes[0].set_xlabel("$q_{early}$",fontsize=18)
    axes[0].set_ylabel("$q_{late}$",fontsize=18)
    axes[0].set_title("homogeneous",fontsize=18)

    axes[1].set_aspect(1)
    cs = axes[1].contourf(x_it,y_it,maskpmf_it.t,levels=np.arange(0,10,1),cmap=cmap)
    axes[1].set_xlabel("$q_{early}$",fontsize=18)
    #axes[1].set_ylabel("$q_{late}$",fontsize=18)
    axes[1].set_yticks([])
    axes[1].set_title("heterogeneous",fontsize=18)
    
    cbar = plt.colorbar(cs)
    cbar.set_label("free energy / $k_bt_f$",fontsize=18)

    fig.subplots_adjust(wspace=0)

    if not os.path.exists("plots"):
        os.mkdir("plots")
    os.chdir("plots")
    plt.savefig("pdz_qearlyvsqlate_0_%d.png" % iteration)
    plt.savefig("pdz_qearlyvsqlate_0_%d.pdf" % iteration)
    plt.savefig("pdz_qearlyvsqlate_0_%d.eps" % iteration)
    plt.show()
    os.chdir("..")


def plot_Q_vs_Qpath(name,iteration,nbins=50):
    os.chdir("%s/iteration_%d" % (name,iteration))

    temps = [ x.rstrip("\n") for x in open("long_temps_last","r").readlines() ]

    for i in range(len(temps)):
        if i == 0:
            qpath = np.loadtxt("%s/qpath.dat" % temps[i])
            Q = np.loadtxt("%s/Q.dat" % temps[i])
        else:
            qpath = np.concatenate((qpath,np.loadtxt("%s/qpath.dat" % temps[i])))
            Q = np.concatenate((Q,np.loadtxt("%s/Q.dat" % temps[i])))

    early = np.loadtxt("early_conts",dtype=int)
    late = np.loadtxt("late_conts",dtype=int)

    n_contacts = len(early) + len(late)

    maxqpath = max(qpath)
    minqpath = min(qpath)
    maxQ = max(Q)
    minQ = min(Q)
     
    Hxy, xedges, yedges = np.histogram2d(Q,qpath,bins=nbins)
    xcenters = 0.5*(xedges[1:] + xedges[:-1])
    ycenters = 0.5*(yedges[1:] + yedges[:-1])
    X,Y = np.meshgrid(xcenters,ycenters)

    pmf = -np.log(Hxy)
    pmf[pmf != np.NaN] -= pmf[pmf != np.NaN].min()

    maskpmf = np.ma.array(pmf,mask=np.isnan(pmf))

    #plt.contourf(X,Y,maskpmf.T)
    plt.contourf(X,Y,maskpmf.T,levels=np.arange(0,10,1))
    plt.xlabel("Folding progress $Q$",fontsize=18)
    plt.ylabel("Pathway $Q_{path}$",fontsize=18)
    
    #plt.ylim(-maxqpath - 1,maxqpath + 1)
    plt.ylim(-50,60)
    cbar = plt.colorbar()
    cbar.set_label("Free energy ($k_BT_f$)",fontsize=18)
    if iteration == 0:
        plt.title("Homogeneous model pathway",fontsize=18)
    else:
        plt.title("Heterogeneous model pathway",fontsize=18)
    plt.savefig("qpathvsQ.pdf")
    plt.savefig("qpathvsQ.png")
    plt.show()

    os.chdir("../..")

def save_legend(name):
    os.chdir("%s/iteration_0" % name)
    
    early = np.loadtxt("early_conts")
    late = np.loadtxt("late_conts")
    
    T = open("long_temps_last","r").readlines()[0].rstrip("\n")
    contacts = np.loadtxt("%s/native_contacts.ndx" % T, skiprows=1)
    n_conts = len(contacts)

    C = np.zeros((n_conts,n_conts),float)
    for i in range(n_conts):
        C[contacts[i,1] - 1,contacts[i,1] - 1] = 0

    os.chdir("../..")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('--name', type=str, required=True, help='Name of protein to plot.')
    parser.add_argument('--iteration', type=int, required=True, help='Iteration to plot.')
    parser.add_argument('--nbins', type=int, default=30, help='Num of bins.')
    args = parser.parse_args()

    name = args.name
    iteration = args.iteration
    nbins = args.nbins

    save_legend(name)

    early = np.loadtxt("%s/iteration_0/early_conts" % name,dtype=int)
    late = np.loadtxt("%s/iteration_0/late_conts" % name,dtype=int)
    n_early = len(early)
    n_late = len(late)
    xbins = np.linspace(0,n_early,nbins)
    ybins = np.linspace(0,n_late,nbins)

    #plot_Q_vs_Qpath(name,iteration,nbins=nbins)
    #plot_Qearly_vs_Qlate(name,iteration,nbins=nbins)
    X0,Y0,maskpmf0,X_it,Y_it,maskpmf_it = get_qearly_qlate(name,iteration,nbins=nbins)

    levels = np.arange(0,10.5,0.5)
    cbarlevels = range(0,11,1)
    
    # plot
    fig = plt.figure(figsize=(13.9,6.125))
    ax0 = fig.add_subplot(1,2,1,aspect="equal")
    ax1 = fig.add_subplot(1,2,2,aspect="equal",sharey=ax0)

    #cmap = plt.get_cmap("hot")
    #cmap = plt.get_cmap("jet")
    #cmap = plt.get_cmap("cool")
    #cmap = plt.get_cmap("gnuplot")
    #cmap = plt.get_cmap("summer")
    #cmap = plt.get_cmap("Blues_r")
    cmap = cubecmap     # With orange at top values
    #cmap = cubeYFcmap

    ax0.set_autoscale_on(False)
    ax0.contourf(X0,Y0,maskpmf0.T,levels=levels,cmap=cmap)
    ax0.set_xlim(0,xbins[-2])
    ax0.set_ylim(0,ybins[-2])
    ax0.set_yticks(range(0,int(round(ybins[-2])),20))
    ax0.set_xticks(range(0,int(round(xbins[-2])),20))
    ax0.text(-10 + n_early/2,-10,"$Q_{early}$",fontsize=20)
    ax0.text(-14,n_late/2,"$Q_{late}$",fontsize=20,rotation="vertical")
    ax0.set_title("Homogeneous",fontsize=20)

    ax1.set_autoscale_on(False)
    cs = ax1.contourf(X_it,Y_it,maskpmf_it.T,levels=levels,cmap=cmap)
    plt.setp(ax1.get_yticklabels(),visible=False)
    ax1.set_xticks(range(20,int(round(xbins[-2])),20))
    ax1.set_xlim(0,xbins[-2])
    ax1.set_ylim(0,ybins[-2])
    ax1.text(-10 + n_early/2,-10,"$Q_{early}$",fontsize=20)
    ax1.set_title("Heterogeneous",fontsize=20)
    
    fig.subplots_adjust(right=0.82)
    cbar_ax = fig.add_axes([0.83, 0.1, 0.025, 0.8])
    cbar = fig.colorbar(cs, cax=cbar_ax)
    cbar.set_label("Free Energy / $k_BT_f$",fontsize=20)
    cbar.set_ticks(cbarlevels)

    fig.subplots_adjust(wspace=0)

    if not os.path.exists("plots"):
        os.mkdir("plots")
    os.chdir("plots")
    plt.savefig("pdz_qearlyvsqlate_0_%d.png" % iteration,dpi=800,bbox_inches='tight')
    plt.savefig("pdz_qearlyvsqlate_0_%d.pdf" % iteration,dpi=800,bbox_inches='tight')
    plt.savefig("pdz_qearlyvsqlate_0_%d.eps" % iteration,dpi=800,bbox_inches='tight')
    plt.show()
    os.chdir("..")
