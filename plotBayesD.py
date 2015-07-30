import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from misc.cube_cmap import cubecmap

def plot_individual_DvsQ(allb2,replicas,name,coord_name,lag_frames,gamma,n_bins):

    binsdir = "%s_diff_model/lag_frames_%d_bins_%d" \
               % (coord_name,lag_frames,n_bins)
    datadir = "%s_diff_model/lag_frames_%d_bins_%d/gamma_%.2e" \
               % (coord_name,lag_frames,n_bins,gamma)

    for n in range(len(allb2)):
        for r in range(len(replicas)):
            os.chdir("random_b2_%s/replica_%d/%s/iteration_0" % (allb2[n],replicas[r],name))
            print "random_b2_%s/replica_%d" % (allb2[n],replicas[r])

            temps = [ x.rstrip("\n") for x in open("long_temps","r").readlines() ]
            T = [ float(x.split("_")[0]) for x in open("long_temps","r").readlines() ]
            if T[0] == T[-1]:
                Tnorm = [ 0.5 for t in T ]
            else:
                Tnorm = [ (t - min(T))/(max(T) - min(T)) for t in T ]


            for i in range(len(temps)):
                print temps[i]
                os.chdir(temps[i])

                bins = np.load("%s/bins.npy" % binsdir)
                bin_centers = 0.5*(bins[1:] + bins[:-1])
                
                F = np.loadtxt("%s/F_final.dat" % datadir)
                D = np.loadtxt("%s/D_final.dat" % datadir)

                plt.figure(1)
                if temps[i].split("_")[1] == "1":
                    plt.plot(bin_centers,F,label=temps[i].split("_")[0],color=cubecmap(Tnorm[i]))
                else:
                    plt.plot(bin_centers,F,color=cubecmap(Tnorm[i]))

                plt.figure(2)
                if temps[i].split("_")[1] == "1":
                    plt.plot(bin_centers,D,label=temps[i].split("_")[0],color=cubecmap(Tnorm[i]))
                else:
                    plt.plot(bin_centers,D,color=cubecmap(Tnorm[i]))
                os.chdir("..")

            if not os.path.exists("plots"):
                os.mkdir("plots")
            os.chdir("plots")

            plt.figure(1)
            plt.xlabel("Q")
            plt.ylabel("F(Q) (k$_B$T)")
            plt.title("%s Free energy lag = %d, n_bins = %d, gamma = %.2e" % (name,lag_frames,n_bins,gamma))
            plt.legend()
            plt.savefig("F_allT.png")
            plt.savefig("F_allT.pdf")

            plt.figure(2)
            plt.xlabel("Q")
            plt.ylabel("D(Q) (Q$^2$/frame)")
            plt.title("%s Diffusion coefficient lag = %d, n_bins = %d, gamma = %.2e" % (name,lag_frames,n_bins,gamma))
            plt.legend(loc=2)
            plt.savefig("D_allT.png")
            plt.savefig("D_allT.pdf")

            #plt.show()
            os.chdir("..")

            os.chdir("../../../..")

def plot_DvsQ_for_all_replicas(allb2,replicas,name,coord_name,lag_frames,n_bins,gamma,n_native_contacts,n_interp=50,units_scaled=False):

    minQ, maxQ = np.loadtxt("min_max_%s_bins" % coord_name,usecols=(1,2),unpack=True)

    #ns_per_frame = 5*(10**-4)
    ps_per_frame = 0.5
    Dscaling = 1./(ps_per_frame*(n_native_contacts**2))

    binsdir = "%s_diff_model/lag_frames_%d_bins_%d" \
               % (coord_name,lag_frames,n_bins)
    datadir = "%s_diff_model/lag_frames_%d_bins_%d/gamma_%.2e" \
               % (coord_name,lag_frames,n_bins,gamma)

    Davg_vs_b = np.zeros((n_interp,len(allb2)))
    #colorbyrep = [ float(i)/9. for i in range(10) ]
    colorbyb2 = [ float(i)/(float(len(allb2)) + 1.) for i in range(len(allb2)) ]

    plt.figure()
    for n in range(len(allb2)):
        interp_points = np.linspace(minQ[n],maxQ[n],n_interp)
        D_replicas = np.zeros((n_interp,len(replicas)))
        print "random_b2_%s" % allb2[n]
        for r in range(len(replicas)):
            os.chdir("random_b2_%s/replica_%d/%s/iteration_0" % (allb2[n],replicas[r],name))
            temps = [ x.rstrip("\n") for x in open("long_temps","r").readlines() ]

            if len(temps) > 3:
                temps = temps[6:9]
            Tf = float(temps[0].split("_")[0]) 


            D_temp = np.zeros((3,n_interp))
            for i in range(len(temps)):
                os.chdir(temps[i])
                bins = np.load("%s/bins.npy" % binsdir)
                bin_centers = 0.5*(bins[1:] + bins[:-1])
                if units_scaled:
                    D = np.loadtxt("%s/D_final.dat" % datadir)*Dscaling
                else:
                    D = np.loadtxt("%s/D_final.dat" % datadir)
                Dfun = interp1d(bin_centers,D)
                D_temp[i,:] = Dfun(interp_points)
                os.chdir("..")

            Davg_temp = np.mean(D_temp,axis=0)
            D_replicas[:,r] = Davg_temp 
            if r == 0:
                if units_scaled:
                    plt.plot(interp_points/n_native_contacts,Davg_temp,label="$b^2=%s$" % allb2[n],lw=2,color=cubecmap(colorbyb2[n]))
                else:
                    plt.plot(interp_points,Davg_temp,label="$b^2=%s$" % allb2[n],lw=2,color=cubecmap(colorbyb2[n]))
            else:
                if units_scaled:
                    plt.plot(interp_points/n_native_contacts,Davg_temp,lw=2,color=cubecmap(colorbyb2[n]))
                else:
                    plt.plot(interp_points,Davg_temp,lw=2,color=cubecmap(colorbyb2[n]))
            os.chdir("../../../..")

        Davg_vs_b[:,n] = np.mean(D_replicas,axis=1)

    plotsdir = "plots/%s_diff_model/lag_frames_%d_bins_%d/gamma_%.2e" % (coord_name,lag_frames,n_bins,gamma)
    if not os.path.exists(plotsdir):
        os.makedirs(plotsdir)
    os.chdir(plotsdir)
    plt.legend(loc=4)
    plt.ylim(0,Davg_vs_b.max()*1.2)
    if units_scaled:
        plt.xlim(0,1.4)
        plt.xlabel("Q Fraction of native contacts ",fontsize=16)
        plt.ylabel("$D(Q)$  (Q$^2$/ps)",fontsize=16)
    else:
        plt.xlim(0,n_native_contacts*1.4)
        plt.xlabel("Q Number of native contacts ",fontsize=16)
        plt.ylabel("$D(Q)$  (Q$^2$/frame)",fontsize=16)
    #plt.title("%s Diffusion coefficient decreases with $b^2$" % name)
    plt.title("%s" % name)

    if units_scaled:
        plt.savefig("DvsQ_allreplicas_nrm.png",bbox_inches="tight")
        plt.savefig("DvsQ_allreplicas_nrm.pdf",bbox_inches="tight")
        plt.savefig("DvsQ_allreplicas_nrm.eps",bbox_inches="tight")
    else:
        plt.savefig("DvsQ_allreplicas.png",bbox_inches="tight")
        plt.savefig("DvsQ_allreplicas.pdf",bbox_inches="tight")
        plt.savefig("DvsQ_allreplicas.eps",bbox_inches="tight")

    plt.show()

    os.chdir("../../../..")

def plot_DvsQ_replica_avg(allb2,replicas,name,coord_name,lag_frames,n_bins,gamma,n_native_contacts,n_interp=50,units_scaled=False):
    #ns_per_frame = 5*(10**-4)
    ps_per_frame = 0.5
    Dscaling = 1./(ps_per_frame*(n_native_contacts**2))
    minQ, maxQ = np.loadtxt("min_max_%s_bins" % coord_name,usecols=(1,2),unpack=True)

    binsdir = "%s_diff_model/lag_frames_%d_bins_%d" \
               % (coord_name,lag_frames,n_bins)
    datadir = "%s_diff_model/lag_frames_%d_bins_%d/gamma_%.2e" \
               % (coord_name,lag_frames,n_bins,gamma)

    Davg_vs_b = np.zeros((n_interp,len(allb2)))
    #colorbyrep = [ float(i)/9. for i in range(10) ]
    colorbyb2 = [ float(i)/(float(len(allb2)) + 1.) for i in range(len(allb2)) ]

    plt.figure()
    for n in range(len(allb2)):
        interp_points = np.linspace(minQ[n],maxQ[n],n_interp)
        D_replicas = np.zeros((n_interp,len(replicas)))
        print "random_b2_%s" % allb2[n]
        for r in range(len(replicas)):
            os.chdir("random_b2_%s/replica_%d/%s/iteration_0" % (allb2[n],replicas[r],name))
            temps = [ x.rstrip("\n") for x in open("long_temps","r").readlines() ]

            if len(temps) > 3:
                temps = temps[6:9]
            Tf = float(temps[0].split("_")[0]) 


            D_temp = np.zeros((3,n_interp))
            for i in range(len(temps)):
                os.chdir(temps[i])
                bins = np.load("%s/bins.npy" % binsdir)
                bin_centers = 0.5*(bins[1:] + bins[:-1])
                if units_scaled:
                    D = np.loadtxt("%s/D_final.dat" % datadir)*Dscaling
                else:
                    D = np.loadtxt("%s/D_final.dat" % datadir)
                Dfun = interp1d(bin_centers,D)
                D_temp[i,:] = Dfun(interp_points)
                os.chdir("..")

            Davg_temp = np.mean(D_temp,axis=0)
            D_replicas[:,r] = Davg_temp 
            os.chdir("../../../..")

        Davg_vs_b[:,n] = np.mean(D_replicas,axis=1)
        if units_scaled:
            plt.plot(interp_points/n_native_contacts,Davg_vs_b[:,n],label="$b^2=%s$" % allb2[n],lw=2,color=cubecmap(colorbyb2[n]))
        else:
            plt.plot(interp_points,Davg_vs_b[:,n],label="$b^2=%s$" % allb2[n],lw=2,color=cubecmap(colorbyb2[n]))

    plotsdir = "plots/%s_diff_model/lag_frames_%d_bins_%d/gamma_%.2e" % (coord_name,lag_frames,n_bins,gamma)
    if not os.path.exists(plotsdir):
        os.makedirs(plotsdir)
    os.chdir(plotsdir)
    plt.legend(loc=4)
    plt.ylim(0,Davg_vs_b.max()*1.2)
    if units_scaled:
        plt.xlim(0,1.4)
        plt.xlabel("Q Fraction of native contacts ",fontsize=16)
        plt.ylabel("Replica averaged $\\overline{D(Q)}$  (Q$^2$/ps)",fontsize=16)
    else:
        plt.xlim(0,n_native_contacts*1.4)
        plt.xlabel("Q Number of native contacts ",fontsize=16)
        plt.ylabel("Replica averaged $\\overline{D(Q)}$  (Q$^2$/frame)",fontsize=16)
    #plt.title("%s Diffusion coefficient decreases with $b^2$" % name)
    plt.title("%s" % name)

    if units_scaled:
        plt.savefig("DvsQ_repavg_nrm.png",bbox_inches="tight")
        plt.savefig("DvsQ_repavg_nrm.pdf",bbox_inches="tight")
        plt.savefig("DvsQ_repavg_nrm.eps",bbox_inches="tight")
    else:
        plt.savefig("DvsQ_repavg.png",bbox_inches="tight")
        plt.savefig("DvsQ_repavg.pdf",bbox_inches="tight")
        plt.savefig("DvsQ_repavg.eps",bbox_inches="tight")

    plt.show()

    os.chdir("../../../..")

def get_coord_min_max(allb2,replicas,name,coord_name,lag_frames,n_bins,gamma):

    binsdir = "%s_diff_model/lag_frames_%d_bins_%d" % (coord_name,lag_frames,n_bins)
     
    min_max_coord = np.zeros((len(allb2),3))
    
    b2 = np.array([ float(x) for x in allb2 ])
    min_max_coord[:,0] = b2

    for n in range(len(allb2)):
        print "random_b2_%s" % allb2[n]
        rep_min = []
        rep_max = []
        for r in range(len(replicas)):
            os.chdir("random_b2_%s/replica_%d/%s/iteration_0" % (allb2[n],replicas[r],name))
            temps = [ x.rstrip("\n") for x in open("long_temps","r").readlines() ]

            if len(temps) > 3:
                temps = temps[6:9]

            for i in range(len(temps)):
                os.chdir(temps[i])
                bins = np.load("%s/bins.npy" % binsdir)
                bin_centers = 0.5*(bins[1:] + bins[:-1])
                rep_min.append(min(bin_centers))
                rep_max.append(max(bin_centers))
                os.chdir("..")
            os.chdir("../../../..")
        min_max_coord[n,1] = max(rep_min) + 2
        min_max_coord[n,2] = min(rep_max) - 2
    np.savetxt("min_max_%s_bins" % coord_name,min_max_coord)

