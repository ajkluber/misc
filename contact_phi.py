import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

def get_sec_structure(name):
    element = []
    bounds = []
    for line in open("%s/secondary_structure.txt" % name,"r"):
        a = line.split()
        element.append(a[0])
        bounds.append([int(a[1]),int(a[2])])
    bounds = np.array(bounds)

    return element, bounds


def get_sim_ddG_phi_res(name,iteration):

    ddG = np.loadtxt("%s/iteration_%d/newton/sim_feature.dat" % (name,iteration))
    
    N = len(ddG)/2
    sim_phi = []
    for i in range(N):
        if ddG[i+N] == 0.:
            sim_phi.append(0)
        else:
            sim_phi.append(ddG[i]/ddG[i+N])
    return np.array(sim_phi)

def get_sim_ddG_phi_ss(name,iteration):

    phi = get_sim_ddG_phi_res(name,iteration)
    exp_phi,indxs = get_exp_phi_res(name)

    element, bounds = get_sec_structure(name)

    phi_ss = []
    #print "Element     Seq. Bounds  Phi"
    output = "# Element     Seq. Bounds  Phi\n"
    for i in range(len(element)):
        ## only average over values that aren't exactly zero.
        residues = ((phi != 0.).astype(int)*(indxs >= bounds[i,0]).astype(int)*(indxs <= bounds[i,1]).astype(int)).astype(bool)
        temp = np.mean(phi[residues])
        phi_ss.append(temp)
        #print "%-9s   %4d %4d   %.4f " % (element[i],bounds[i,0],bounds[i,1],temp)
        output += "%-9s   %4d %4d   %.4f \n" % (element[i],bounds[i,0],bounds[i,1],temp)

    outfile = "%s/iteration_%d/ddG_phi_ss" % (name,iteration)
    if not os.path.exists(outfile):
        open(outfile,"w").write(output)

    return phi_ss

def calculate_contact_phi_res(name,iteration):
    ''' Calculate the percentage of native structure for each residue.'''
    pdb = open("%s/clean.pdb" % name,"r").readlines()
    n_residues = len(open("%s/Native.pdb" % name,"r").readlines()) - 1

    cwd = os.getcwd()
    sub =  "%s/iteration_%d" % (name,iteration)
    os.chdir(sub)

    if not os.path.exists("contact_phi_res"):
        Tuse = open("long_temps_last","r").readlines()[0].rstrip("\n")
        if not os.path.exists("%s/contacts.ndx" % Tuse):
            contacts = np.loadtxt("%s/native_contacts.ndx" % Tuse,skiprows=1,dtype=int)
        else:
            contacts = np.loadtxt("%s/contacts.ndx" % Tuse,skiprows=1,dtype=int)

        #C = np.zeros((n_residues,n_residues),float)
        U  = np.loadtxt("cont_prob_%s.dat" % "U")
        TS = np.loadtxt("cont_prob_%s.dat" % "TS")
        N  = np.loadtxt("cont_prob_%s.dat" % "N")

        phi_res = np.zeros(n_residues,float)
        phi_contact = (TS - U)/(N - U)
        for i in range(n_residues):
            ## Average over contact formation for all contacts that residue i participates in.
            contacts_for_residue = ((contacts[:,0] == (i+1)).astype(int) + (contacts[:,1] == (i+1)).astype(int)).astype(bool)
            if np.any(contacts_for_residue):
                phi_res[i] = np.mean(phi_contact[contacts_for_residue])

        np.savetxt("contact_phi_res", phi_res)
    else:
        phi_res = np.loadtxt("contact_phi_res")

    if not os.path.exists("%s_contact_phi.pdb" % name):
        newpdb = ""
        for line in pdb:
            if line.startswith("END"):
                newpdb += "END"
                break
            else:
                resnum = int(line[22:26]) - 1
                newpdb += "%s     %5f\n" % (line[:55],phi_res[resnum])

        open("%s_contact_phi.pdb" % name,"w").write(newpdb)
    os.chdir(cwd)

    return phi_res


def calculate_average_contact_phi_ss(name,iteration):

    phi = calculate_contact_phi_res(name,iteration)

    element, bounds = get_sec_structure(name)

    indxs = np.arange(len(phi))
    phi_ss = []
    #print "Element     Seq. Bounds  Phi"
    output = "# Element     Seq. Bounds  Phi\n"
    for i in range(len(element)):
        ## only average over values that aren't exactly zero.
        residues = ((phi != 0.).astype(int)*(indxs >= (bounds[i,0] - 1)).astype(int)*(indxs <= (bounds[i,1] - 1)).astype(int)).astype(bool)
        #temp = np.mean(phi[bounds[i,0]:bounds[i,1]])
        temp = np.mean(phi[residues])
        phi_ss.append(temp)
        #print "%-9s   %4d %4d   %.4f " % (element[i],bounds[i,0],bounds[i,1],temp)
        output += "%-9s   %4d %4d   %.4f \n" % (element[i],bounds[i,0],bounds[i,1],temp)

    outfile = "%s/iteration_%d/contact_phi_ss" % (name,iteration)
    if not os.path.exists(outfile):
        open(outfile,"w").write(output)

    return phi_ss

def get_exp_phi_res(name):
    lines = [ x.rstrip("\n") for x in open("%s/mutants/core.ddG" % name,"r").readlines() ]
    exp_phi_res = []
    indxs = []

    for line in lines[1:]:
        vals = line.split()
        if int(vals[8]) == 1:
            continue
        else:
            phi = float(vals[7])
            mut_idx = int(vals[0])
            indxs.append(mut_idx)
            if phi == -999.:
                exp_phi_res.append(0)
            else:
                exp_phi_res.append(phi)

    return exp_phi_res,np.array(indxs)

def get_exp_phi_ss(name):
    element, bounds = get_sec_structure(name)

    lines = [ x.rstrip("\n") for x in open("%s/mutants/core.ddG" % name,"r").readlines() ]

    exp_phi_ss = [ [] for i in range(len(element)) ]

    for line in lines[1:]:
        vals = line.split()
        if int(vals[8]) == 1:
            continue
        else:
            mut_idx = int(vals[0])
            for j in range(len(element)):
                if (mut_idx >= bounds[j][0]) and (mut_idx <= bounds[j][1]):
                    exp_phi_ss[j].append(float(vals[7]))
                    break

    #print exp_phi_ss    ## DEBUGGIN
    avg_phi_ss = np.zeros(len(element))
    for j in range(len(element)):
        avg_phi_ss[j] = np.mean(exp_phi_ss[j])

    return avg_phi_ss,element

def plot_exp_sim_contact_phi_ss(name,iteration):
    sim_phi_ss = calculate_average_contact_phi_ss(name,iteration)
    exp_phi_ss,element = get_exp_phi_ss(name) 

    ## Calculate least squares fit and R^2 value.
    x = np.array(sim_phi_ss)
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A,exp_phi_ss)[0]
    fit = m*x + c*np.ones(len(x))

    SEy = np.sum((exp_phi_ss - np.mean(exp_phi_ss))**2)
    SEline = np.sum((exp_phi_ss - fit)**2)
    r2 = 1. - (SEline/SEy)
     
    ## Plot
    plt.figure()
    plt.plot([0,1],[0,1],'k',lw=2)
    #plt.plot(x,fit,'b',lw=2,label="y = %.3fx + %.3f  R$^2$=%.3f" % (m,c,r2))
    plt.plot(x,fit,'b',lw=2,label="R$^2$=%.3f" % r2)
    for i in range(len(element)):
        plt.plot(sim_phi_ss[i],exp_phi_ss[i],'o',label=element[i])
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.grid(True)
    plt.legend(loc=4)
    plt.xlabel("simulation $\phi_{ss}$")
    plt.ylabel("experiment $\phi_{ss}$")
    plt.title("$\phi_{ss}$ comparison for %s iteration %d" % (name,iteration))
    if not os.path.exists("%s/iteration_%d/plots" % (name,iteration)):
        os.mkdir("%s/iteration_%d/plots" % (name,iteration))
    plt.savefig("%s/iteration_%d/plots/compare_contact_phi_ss.png" % (name,iteration))
    plt.savefig("%s/iteration_%d/plots/compare_contact_phi_ss.pdf" % (name,iteration))
    #plt.show()

def plot_exp_sim_contact_phi_res(name,iteration):
    exp_phi_res,mut_indxs = get_exp_phi_res(name) 
    sim_phi = calculate_contact_phi_res(name,iteration)

    #print mut_indxs
    #raise SystemExit
    sim_phi_res = sim_phi[mut_indxs - 1]

    ## Calculate least squares fit and R^2 value.
    x = np.array(sim_phi_res)
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A,exp_phi_res)[0]
    fit = m*x + c*np.ones(len(x))

    SEy = np.sum((exp_phi_res - np.mean(exp_phi_res))**2)
    SEline = np.sum((exp_phi_res - fit)**2)
    r2 = 1. - (SEline/SEy)
     
    ## Plot
    plt.figure()
    plt.plot([0,1],[0,1],'k',lw=2)
    #plt.plot(x,fit,'b',lw=2,label="y = %.3fx + %.3f  R$^2$=%.3f" % (m,c,r2))
    plt.plot(x,fit,'b',lw=2,label="R$^2$=%.3f" % r2)
    for i in range(len(exp_phi_res)):
        plt.plot(sim_phi_res[i],exp_phi_res[i],'o')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.grid(True)
    plt.legend(loc=4)
    plt.xlabel("simulation $\phi$")
    plt.ylabel("experiment $\phi$")
    plt.title("$\phi$ from contacts for %s iteration %d" % (name,iteration))
    if not os.path.exists("%s/iteration_%d/plots" % (name,iteration)):
        os.mkdir("%s/iteration_%d/plots" % (name,iteration))
    plt.savefig("%s/iteration_%d/plots/compare_contact_phi_res.png" % (name,iteration))
    plt.savefig("%s/iteration_%d/plots/compare_contact_phi_res.pdf" % (name,iteration))
    #plt.show()

def plot_exp_sim_ddG_phi_res(name,iteration):
    exp_phi_res,mut_indxs = get_exp_phi_res(name) 
    sim_phi_res = get_sim_ddG_phi_res(name,iteration)
    
    ## Calculate least squares fit and R^2 value.
    x = np.array(sim_phi_res)
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A,exp_phi_res)[0]
    fit = m*x + c*np.ones(len(x))

    SEy = np.sum((exp_phi_res - np.mean(exp_phi_res))**2)
    SEline = np.sum((exp_phi_res - fit)**2)
    r2 = 1. - (SEline/SEy)
     
    ## Plot
    plt.figure()
    plt.plot([0,1],[0,1],'k',lw=2)
    #plt.plot(x,fit,'b',lw=2,label="y = %.3fx + %.3f  R$^2$=%.3f" % (m,c,r2))
    plt.plot(x,fit,'b',lw=2,label="R$^2$=%.3f" % r2)
    for i in range(len(sim_phi_res)):
        plt.plot(sim_phi_res[i],exp_phi_res[i],'o')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.grid(True)
    plt.legend(loc=4)
    plt.xlabel("simulation $\phi$")
    plt.ylabel("experiment $\phi$")
    plt.title("$\phi$ from ddG's for %s iteration %d" % (name,iteration))
    if not os.path.exists("%s/iteration_%d/plots" % (name,iteration)):
        os.mkdir("%s/iteration_%d/plots" % (name,iteration))
    plt.savefig("%s/iteration_%d/plots/compare_ddG_phi_res.png" % (name,iteration))
    plt.savefig("%s/iteration_%d/plots/compare_ddG_phi_res.pdf" % (name,iteration))
    #plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('--name', type=str, required=True, help='Name of protein to plot.')
    parser.add_argument('--iteration', type=int, required=True, help='Iteration to plot.')
    args = parser.parse_args()

    name = args.name
    iteration = args.iteration

    plot_exp_sim_contact_phi_ss(name,iteration)
    plot_exp_sim_contact_phi_res(name,iteration)
    plot_exp_sim_ddG_phi_res(name,iteration)
    plt.show()
