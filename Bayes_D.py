""" Estimate free energy and diffusion coefficient along reaction coordinate

Description
-----------
    Estimate the free energy F(x) and diffusion coefficient D(x) for
Smoluchowski diffusion along a reaction coordinate x. The likelihood of
observing some given reaction coordinate dynamics can be calculated
as the product of propagator matrix elements.

The propagator of the dynamics, of fixed lagtime dt, is:
P(j, t + dt| i, t) = prob. of hopping from bin i to bin j in time dt


"""

import logging
import argparse
import os
import numpy as np
from scipy import linalg

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import misc.hummer as hummer

def attempt_step_D(beta_MC,neg_lnL,D,F,D_step,t_alpha,Nij,n_bins,D_attempts,D_accepts,dx,gamma,step_scale):
    """Attempt a monte carlo step in D, the diffusion coefficients"""

    which_bin = np.random.randint(n_bins)
    D_trial = np.array(D,copy=True)
    #step = np.sign(np.random.rand() - 0.5)*D_step[which_bin]
    step = np.random.normal(scale=step_scale)
    if (D_trial[which_bin] + step) <= 0:
        # Step rejected. We don't let D be negative
        pass
    else:
        D_trial[which_bin] += step

        Propagator = hummer.calculate_propagator(n_bins,F,D_trial,dx,t_alpha)
        neg_lnL_trial = hummer.calculate_logL_smoothed_FD(n_bins,Nij,Propagator,D_trial,gamma)
        
        if neg_lnL_trial < neg_lnL:
            # Accept the move if it increases the log Likelihood function.
            D = D_trial
            neg_lnL = neg_lnL_trial
            D_accepts[which_bin] += 1
        else:
            delta_lnL = neg_lnL_trial - neg_lnL
            if np.random.rand() <= np.exp(-beta_MC*delta_lnL):
                # Step accepted
                D = D_trial
                neg_lnL = neg_lnL_trial
                D_accepts[which_bin] += 1
            else:
                # Step rejected
                pass

    D_attempts[which_bin] += 1

    return neg_lnL,D

def attempt_step_F(beta_MC,neg_lnL,D,F,F_step,t_alpha,Nij,n_bins,F_attempts,F_accepts,dx,gamma,step_scale):
    """Attempt a monte carlo step in Rij, the rate matrix"""

    which_bin = np.random.randint(n_bins)
    F_trial = np.array(F,copy=True)
    #step = np.sign(np.random.rand() - 0.5)*F_step[which_bin]
    step = np.random.normal(scale=step_scale)

    F_trial[which_bin] += step

    Propagator = hummer.calculate_propagator(n_bins,F_trial,D,dx,t_alpha)
    neg_lnL_trial = hummer.calculate_logL_smoothed_FD(n_bins,Nij,Propagator,D,gamma)
    
    if neg_lnL_trial < neg_lnL:
        # Accept the move if it increases the log Likelihood function.
        F = F_trial
        neg_lnL = neg_lnL_trial
        F_accepts[which_bin] += 1
    else:
        delta_lnL = neg_lnL_trial - neg_lnL
        if np.random.rand() <= np.exp(-beta_MC*delta_lnL):
            # Step accepted
            F = F_trial
            neg_lnL = neg_lnL_trial
            F_accepts[which_bin] += 1
        else:
            # Step rejected
            pass

    F_attempts[which_bin] += 1

    return neg_lnL,F

def calculate_propagator(n_bins,F,D,dx,t_alpha):
    """Calculate propagator analytically"""
    omega = lambda i,j: 0.5*((D[i] + D[j])/(dx**2))*np.exp(-0.5*(F[j] - F[i]))

    M = np.zeros((n_bins,n_bins))  
    for i in range(1,n_bins - 1):
        M[i,i] = -(omega(i,i + 1) + omega(i,i - 1))
        M[i,i + 1] = np.sqrt(omega(i,i + 1)*omega(i + 1,i))
        M[i + 1,i] = np.sqrt(omega(i + 1,i)*omega(i,i + 1))
        M[i,i - 1] = np.sqrt(omega(i,i - 1)*omega(i - 1,i))
        M[i - 1,i] = np.sqrt(omega(i - 1,i)*omega(i,i - 1))
                                 
    M[0,0] = -omega(0,1)
    M[n_bins - 1,n_bins - 1] = -omega(n_bins - 1,n_bins - 2)

    S,V = np.linalg.eig(M)
    Propagator = abs(np.dot(V,np.dot(np.diag(np.exp(S.real*t_alpha)),V.T)))
    return Propagator

def plot_and_save(neg_lnL_all,F_all,D_all,F,D,
                bin_centers,beta_MC_schedule,beta_MC_steps,
                n_stages,n_attempts,gamma,coord_name,no_display,save=True):

    neg_lnL_all = np.array(neg_lnL_all)
    neg_lnL_all /= neg_lnL_all[0]
    D_all = np.array(D_all)
    F_all = np.array(F_all)

    logging.info("Saving:")
    logging.info("  annealing_schedule")
    logging.info("  F_final.dat D_final.dat")
    logging.info("  F_all.npy D_all.npy")
    logging.info("  neg_lnL_all.npy")
    if save:
        with open("annealing_schedule","w") as fout:
            fout.write("#Beta  n_steps\n")
            for i in range(len(beta_MC_schedule)):
                fout.write("%.5f  %d\n" % (beta_MC_schedule[i],beta_MC_steps[i]))
            
        np.savetxt("F_final.dat",F)
        np.savetxt("D_final.dat",D)
        np.save("F_all.npy",F_all)
        np.save("D_all.npy",D_all)
        np.save("neg_lnL_all.npy",neg_lnL_all)

    logging.info("Plotting: (everything as png & pdf)")
    logging.info("  lnL")
    logging.info("  F_final  F_all")
    logging.info("  D_final  D_all")
    plt.figure()
    plt.plot(neg_lnL_all)
    sum_steps = 0
    for i in range(n_stages):
        sum_steps += beta_MC_steps[i]*n_attempts
        plt.axvline(x=sum_steps,color='k')
        plt.text(sum_steps - 0.8*beta_MC_steps[i]*n_attempts,
                0.9*(max(neg_lnL_all) - min(neg_lnL_all)) + min(neg_lnL_all),
                "$\\beta=%.1e$" % beta_MC_schedule[i],fontsize=16)
    plt.title("Negative log likelihood",fontsize=16)
    if save:
        plt.savefig("lnL.png")
        plt.savefig("lnL.pdf")

    plt.figure()
    plt.plot(D_all)
    sum_steps = 0
    for i in range(n_stages):
        sum_steps += beta_MC_steps[i]*n_attempts
        plt.axvline(x=sum_steps,color='k')
    plt.xlabel("MC steps",fontsize=16)
    plt.title("Diffusion coefficient $\\gamma=%.2e$" % gamma,fontsize=16)
    if save:
        plt.savefig("D_all.png")
        plt.savefig("D_all.pdf")

    plt.figure()
    plt.plot(F_all)
    sum_steps = 0
    for i in range(n_stages):
        sum_steps += beta_MC_steps[i]*n_attempts
        plt.axvline(x=sum_steps,color='k')
    plt.xlabel("MC steps",fontsize=16)
    plt.title("Free energy $\\gamma=%.2e$" % gamma,fontsize=16)
    if save:
        plt.savefig("F_all.png")
        plt.savefig("F_all.pdf")

    plt.figure()
    plt.plot(bin_centers,F)
    plt.xlabel("%s" % coord_name,fontsize=16)
    plt.ylabel("F(%s) (k$_B$T)" % coord_name,fontsize=16)
    plt.title("Final Free energy $\\gamma=%.2e$" % gamma,fontsize=16)
    if save:
        plt.savefig("F_final.png")
        plt.savefig("F_final.pdf")

    plt.figure()
    plt.plot(bin_centers,D)
    plt.xlabel("%s" % coord_name,fontsize=16)
    plt.ylabel("D(%s)" % coord_name,fontsize=16)
    plt.title("Final Diffusion coefficient $\\gamma=%.2e$" % gamma,fontsize=16)
    if save:
        plt.savefig("D_final.png")
        plt.savefig("D_final.pdf")

    if not no_display:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bayesian estimation of 1D diffusion model.")
    parser.add_argument("--coord_file", 
                        type=str, 
                        required=True,
                        help="Name of reaction coordinate file.")

    parser.add_argument("--lag_frames", 
                        type=int, 
                        required=True, 
                        help="Number of frames to subsample.")

    parser.add_argument("--n_bins", 
                        type=int, 
                        required=True, 
                        help="Number of bins along reaction coordinate.")

    parser.add_argument("--gamma", 
                        type=float, 
                        required=True, 
                        help="Smootheness scale of D. Recommended: 0.001.")

    parser.add_argument("--dt", 
                        type=float, 
                        default=1, 
                        help="Timestep units per frame. Default: 1ps per frame.")

    parser.add_argument("--adaptive_stepsize", 
                        action="store_true",
                        help="Adaptively scale monte carlo stepsize based on acceptance ratio.")

    parser.add_argument("--no_display", 
                        action="store_true",
                        help="Plot things without display available (e.g. on compute node).")

    parser.add_argument("--debug", 
                        action="store_true",
                        help="Plot things without display available (e.g. on compute node).")

    args = parser.parse_args()

    coord_file = args.coord_file
    coord_name = coord_file.split(".")[0]
    file_ext = coord_file.split(".")[-1]
    lag_frames = args.lag_frames
    dt = args.dt
    gamma = args.gamma
    n_bins = args.n_bins
    no_display = args.no_display
    adaptive_stepsize = args.adaptive_stepsize
    debug = args.debug

    t_alpha = lag_frames*dt
    n_attempts = n_bins*2

    run_directory = "%s_diff_model/lag_frames_%d_bins_%d/gamma_%.2e" \
                  % (coord_name,lag_frames,n_bins,gamma)
    logfilename = "%s/Bayes_FD.log" % run_directory

    if not os.path.exists(run_directory):
        os.makedirs(run_directory)

    logging.basicConfig(filename=logfilename,
                        filemode="w",
                        format="%(levelname)s:%(name)s:%(asctime)s: %(message)s",
                        datefmt="%H:%M:%S",
                        level=logging.DEBUG)

    logging.info("Bayesian estimation of 1D diffusion model")
    logging.info("Parameters:")
    logging.info(" coord_file = %s" % coord_file)
    logging.info(" lag_frames = %5d" % lag_frames)
    logging.info(" n_bins     = %5d" % n_bins)
    logging.info(" gamma      = %5.2e" % gamma)
    logging.info(" dt         = %5.2e" % dt)
    logging.info(" adaptive   = %s" % str(adaptive_stepsize))
    logging.info(" no-display = %s" % str(no_display))


    os.chdir("%s_diff_model/lag_frames_%d_bins_%d" % (coord_name,lag_frames,n_bins))
    if os.path.exists("Nij.npy"):
        # Loading transition counts between bins.
        logging.info("Loading transition counts")
        Nij = np.load("Nij.npy")
        bins = np.load("bins.npy")
        bin_centers = 0.5*(bins[1:] + bins[:-1])
        dx = bins[1] - bins[0] 
    else:
        # If bins counts have not been created. Calculate them. 
        logging.info("Calculating transition counts")
        if file_ext == "npy": 
            xfull = np.load("../../%s" % coord_file)
        else:
            xfull = np.loadtxt("../../%s" % coord_file)
        x = xfull[::lag_frames]
        Nx,bins = np.histogram(x,bins=n_bins)
        n_frames = len(x)

        Nij_raw = hummer.count_transitions(n_bins,n_frames,bins,x)
        Nij = 0.5*(Nij_raw + Nij_raw.T)
        np.save("Nij.npy",Nij)
        np.save("bins.npy",bins)

        bin_centers = 0.5*(bins[1:] + bins[:-1])
        dx = bins[1] - bins[0] 

    os.chdir("gamma_%.2e" % gamma)

    ########################################################################
    # Initialize F, D, lnL
    ########################################################################
    logging.info("Initializing F, D, and log-likelihood")
    F = np.ones(n_bins)
    #D = 0.01*np.ones(n_bins)
    D = np.ones(n_bins)

    F_step = 0.01*np.ones(n_bins)
    F_attempts = np.zeros(n_bins)
    F_accepts = np.zeros(n_bins)

    D_step = 0.01*np.ones(n_bins,float)
    D_attempts = np.zeros(n_bins,float)
    D_accepts = np.zeros(n_bins,float)

    Propagator = hummer.calculate_propagator(n_bins,F,D,dx,t_alpha)
    neg_lnL = hummer.calculate_logL_smoothed_FD(n_bins,Nij,Propagator,D,gamma)

    neg_lnL_all = [neg_lnL]
    D_all = [D]
    F_all = [F]

    ########################################################################
    # Perform Metropolis-Hastings monte carlo 
    ########################################################################
    beta_MC_schedule = [2.,3.]          
    #beta_MC_schedule = [0.01,0.02]          
    beta_MC_steps = [200,100]
    D_step_scale = [0.2,0.01]
    F_step_scale = [0.01,0.005]
    n_stages = len(beta_MC_schedule)    
    logging.info("Starting Monte Carlo optimization of Likelihood L")
    for b in range(n_stages):
        # Annealing stage for MC acceptance ratio
        F_scale = F_step_scale[b]
        D_scale = D_step_scale[b]
        beta_MC = float(beta_MC_schedule[b])
        n_steps = beta_MC_steps[b]
        logging.info("Stage %3d of %3d: Beta = %.4e  Total steps = %d" % (b + 1,n_stages,beta_MC,n_steps*n_attempts))
        logging.info("  Step #      -log(L)")
        if debug:
            print "Stage %3d of %3d: Beta = %.4e  Total steps = %d" % (b + 1,n_stages,beta_MC,n_steps*n_attempts)
            print "  Step #      -log(L)"
        for n in range(n_steps):
            for i in range(n_attempts):
                neg_lnL,D = hummer.attempt_step_D(beta_MC,neg_lnL,D,F,D_step,t_alpha,Nij,n_bins,D_attempts,D_accepts,dx,gamma,D_scale)
                neg_lnL,F = hummer.attempt_step_F(beta_MC,neg_lnL,D,F,F_step,t_alpha,Nij,n_bins,F_attempts,F_accepts,dx,gamma,F_scale)
                neg_lnL_all.append(neg_lnL)
                D_all.append(D)
                F_all.append(F)

            logging.info("  %-10d  %-15.4f" % (n*n_attempts,neg_lnL))
            if debug:
                print "  %-10d  %-15.4f" % (n*n_attempts,neg_lnL)

            if adaptive_stepsize: 
                # Adaptively scale step size to match acceptance
                # ratio of 0.5 in all bins. Use with caution.
                if np.all(F_attempts):
                    ratio_F = F_accepts/F_attempts
                    F_step[ratio_F <= 0.5] *= 0.95
                    F_step[ratio_F > 0.5] *= 1.05
                if np.all(D_attempts):
                    ratio_D = D_accepts/D_attempts
                    D_step[ratio_D <= 0.5] *= 0.95
                    D_step[ratio_D > 0.5] *= 1.05
    
    plot_and_save(neg_lnL_all,F_all,D_all,F,D,
                bin_centers,beta_MC_schedule,beta_MC_steps,
                n_stages,n_attempts,gamma,coord_name,no_display,save=False)

    logging.info("Done")
    os.chdir("../../..")
