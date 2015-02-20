""" Bayesian estimates of F(Q) and D(Q)



Ingredients:
  1. Reaction coordinate Q.
  2. Equilibrium trajectory Q(t)
  3. 

Procedure: 
    1. Assume uniform prior distribution of equilibrium probabili



1. Read in coordinate. Given n_bins calculate bin_width and Nij transition
    counts.
2. Initialize -lnL with smoothening prior. g = -lnP, Rij = rate coefficient
    matrix.
3. Monte Carlo loop
    For step in n_steps:
        For attempt in n_attempts:
            Attempt a step in g 
            Attempt a step in R

        Adjust step size of delta_g, delta_R in order to bring acceptance ratio
        closer to 0.5

4. Do another loop to determine the error bars


"""

import os
import numpy as np
from scipy import linalg

import matplotlib.pyplot as plt
import matplotlib

#from hummer_2005 import *
import hummer_2005

def imshow_mask_zero(N):
    # Plot matrix but mask zeros.
    M = np.array(N,copy=True) 
    M[M == 0] = np.NaN
    jet = matplotlib.cm.jet
    temp = np.ma.array(M,mask=np.isnan(M))
    jet.set_bad('w',1.)
    plt.imshow(temp,cmap=jet,interpolation='none')
    plt.colorbar()
    plt.show()

def attempt_step_g(neg_lnL,g,g_step,Rij,Nij,P,beta,t_alpha,n_bins,g_accepts,g_attempts,gamma,deltaQ):
    """Attempt a monte carlo step in g, the free energy of bins"""

    which_bin = np.random.randint(n_bins)
    g_trial = np.array(g,copy=True)
    g_trial[which_bin] += np.sign(np.random.rand() - 0.5)*g_step[which_bin]

    P_trial = deltaQ*np.exp(-beta*g_trial)
    P_trial /= sum(P_trial)

    Rij_trial = hummer_2005.calculate_detailed_balance_R(n_bins,Rij,P_trial)
    
    exp_tRij = linalg.expm(t_alpha*Rij_trial)
    neg_lnL_trial = hummer_2005.calculate_logL_smoothed(n_bins,deltaQ,gamma,Nij,Rij_trial,exp_tRij,P_trial)
    
    if neg_lnL_trial < neg_lnL:
        # Accept the move if it increases the log Likelihood function.
        g = g_trial
        P = P_trial
        Rij = Rij_trial
        neg_lnL = neg_lnL_trial
        g_accepts[which_bin] += 1
        g_attempts[which_bin] += 1
    else:
        delta_lnL = neg_lnL_trial - neg_lnL
        if np.random.rand() <= np.exp(-delta_lnL):
            # Step accepted
            g = g_trial
            P = P_trial
            Rij = Rij_trial
            neg_lnL = neg_lnL_trial
            g_accepts[which_bin] += 1
            g_attempts[which_bin] += 1
        else:
            # Step rejected
            g_attempts[which_bin] += 1

    return neg_lnL,Rij,P,g

def attempt_step_Rij(neg_lnL,Rij,R_step,Nij,P,t_alpha,beta,n_bins,gamma,deltaQ):
    """Attempt a monte carlo step in Rij, the rate matrix"""

    # attempt a step in R
    which_bin = np.random.randint(n_bins - 1)
    Rij_trial = np.array(Rij,copy=True)
    Rij_trial[which_bin + 1,which_bin] += \
        np.sign(np.random.rand() - 0.5)*np.random.rand()*R_step[which_bin]

    Rij_trial = hummer_2005.calculate_detailed_balance_R(n_bins,Rij_trial,P)

    exp_tRij = linalg.expm(t_alpha*Rij_trial)
    neg_lnL_trial = hummer_2005.calculate_logL_smoothed(n_bins,deltaQ,gamma,Nij,Rij,exp_tRij,P)

    if neg_lnL_trial < neg_lnL:
        # Accept the move if it increases the log Likelihood function.
        Rij = Rij_trial
        neg_lnL = neg_lnL_trial
        R_accepts[which_bin] += 1
        R_attempts[which_bin] += 1
    else:
        delta_lnL = neg_lnL_trial - neg_lnL
        if np.random.rand() <= np.exp(-delta_lnL):
            # Step accepted
            Rij = Rij_trial
            neg_lnL = neg_lnL_trial
            R_accepts[which_bin] += 1
            R_attempts[which_bin] += 1
        else:
            # Step rejected
            R_attempts[which_bin] += 1

    return neg_lnL,Rij,P

if __name__ == "__main__":

    n_attempts = 50
    n_equil_steps = 200
    n_steps = 500
    n_bins = 25

    gamma = 0.001 # Scale over which diffusion coefficient should be smooth. Depends on coordinate
    delta_t = 0.5
    timelag = 500 # 500001*3 total frames
    t_alpha = delta_t*timelag 

    GAS_CONSTANT_KJ_MOL = 0.0083145
    T = float(open("temp.dat","r").read().rstrip("\n"))
    beta = 1./(GAS_CONSTANT_KJ_MOL*T)

    # Assume Q has been normalized already
    Qfull = np.loadtxt("Qnrm.dat")

    Q = Qfull[::timelag]
    n_frames = len(Q)

    bins = np.linspace(min(Q),max(Q),num=n_bins+1)
    deltaQ = bins[1] - bins[0]

    #############################################################################
    # Initialize D,g,Nij,Rij
    #############################################################################
    print "Estimating diffusion model using: timelag = %d  n_bins = %d " % (timelag,n_bins)
    if os.path.exists("Nij_%d.dat" % n_bins):
        Nij = np.loadtxt("Nij_%d.dat" % n_bins)
    else:
        print "Count observed transitions between bins"
        Nij = hummer_2005.count_transitions(n_bins,n_frames,bins,Q)
        np.savetxt("Nij_%d.dat" % n_bins,Nij)

    #imshow_mask_zero(Nij) 

    print "Initializing guess for P_i, R_ij"
    # Propability of each bin and free energy g.
    P = 0.3*np.ones(n_bins,float)
    P /= sum(P)
    g = -np.log(P)

    # Rij must satisfy detailed balance.
    Rij = np.zeros((n_bins,n_bins),float)

    for i in range(1,n_bins):
        Rij[i,i - 1] = 0.3

    Rij = hummer_2005.calculate_detailed_balance_R(n_bins,Rij,P)

    #############################################################################
    # Calculate smoothened log-likelihood given D,P,Nij,Rij
    #############################################################################

    exp_tRij = linalg.expm(t_alpha*Rij)
    neg_lnL = hummer_2005.calculate_logL_smoothed(n_bins,deltaQ,gamma,Nij,Rij,exp_tRij,P)

    #raise SystemExit
    #print neg_lnL

    #############################################################################
    # Monte Carlo optimization of g and Rij. Equilibration loop.
    #############################################################################
    g_accepts = np.zeros(n_bins,float)
    R_accepts = np.zeros(n_bins,float)
    g_attempts = np.zeros(n_bins,float)
    R_attempts = np.zeros(n_bins,float)

    g_step = np.random.rand(n_bins)
    R_step = np.random.rand(n_bins - 1)
    print "Running %d equilibration steps:" % n_equil_steps
    for n in range(n_equil_steps):
        for x in range(n_attempts+np.random.randint(50)):
            neg_lnL,Rij,P,g = attempt_step_g(neg_lnL,g,g_step,Rij,Nij,P,beta,t_alpha,n_bins,g_accepts,g_attempts,gamma,deltaQ)
            neg_lnL,Rij,P = attempt_step_Rij(neg_lnL,Rij,R_step,Nij,P,t_alpha,beta,n_bins,gamma,deltaQ)

        if np.all(g_attempts):
            ratio_g = g_accepts/g_attempts
            g_step += ((ratio_g - 0.5)**9)*g_step
        if np.all(R_attempts):
            ratio_R = R_accepts/R_attempts
            R_step += ((ratio_R - 0.5)**9)*R_step
        
        if (n % 10) == 0:
            print " %5d  %20.4f" % (n,neg_lnL)

    #############################################################################
    # Monte Carlo optimization of g and Rij. Production runn.
    #############################################################################

    F_all = np.zeros((n_steps,n_bins),float)
    D_all = np.zeros((n_steps,n_bins-1),float)
    print "Running %d production steps:" % n_steps
    for n in range(n_steps):
        for x in range(n_attempts+np.random.randint(50)):
            neg_lnL,Rij,P,g = attempt_step_g(neg_lnL,g,g_step,Rij,Nij,P,beta,t_alpha,n_bins,g_accepts,g_attempts,gamma,deltaQ)
            neg_lnL,Rij,P = attempt_step_Rij(neg_lnL,Rij,R_step,Nij,P,t_alpha,beta,n_bins,gamma,deltaQ)

        if np.all(g_attempts):
            ratio_g = g_accepts/g_attempts
            g_step += ((ratio_g - 0.5)**9)*g_step
        if np.all(R_attempts):
            ratio_R = R_accepts/R_attempts
            R_step += ((ratio_R - 0.5)**9)*R_step
        
        if (n % 10) == 0:
            print " %5d  %20.4f" % (n,neg_lnL)

        F_all[n,:] = -np.log(P/deltaQ)
        #D = np.zeros(n_bins-1,float)
        for i in range(n_bins-1):
            D_all[n,i] = (deltaQ**2)*Rij[i + 1,i]*np.sqrt(P[i]/P[i + 1])


    exp_tRij = linalg.expm(t_alpha*Rij)

    if not os.path.exists("lag_time_%d_bins_%d" % (timelag,n_bins)):
        os.mkdir("lag_time_%d_bins_%d" % (timelag,n_bins))

    os.chdir("lag_time_%d_bins_%d" % (timelag,n_bins))
    np.savetxt("expRij.dat",exp_tRij)
    np.savetxt("P.dat",P)
    np.savetxt("F_all.dat",F_all)
    np.savetxt("D_all.dat",D_all)
    np.savetxt("Qbins.dat",bins)
    os.chdir("..")

