import os
import sys
import pdb
import time
import argparse
import logging 
import numpy as np

from memory_profiler import profile
import mdtraj as md

# Script utility for the calculation of contact based reaction coordinates

######################################################################
# Utility functions
######################################################################
def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def check_traj_lengths(dirs):
    """Get trajectory lengths if possible"""
    n_traj = len(dirs)
    traj_lengths = [] 
    if all([ os.path.exists("%s/n_frames" % dirs[i]) for i in range(n_traj) ]):
        for i in range(n_traj):
            with open("%s/n_frames" % dirs[i],"r") as fin:
                n_frms = int(fin.read().rstrip("\n"))
            traj_lengths.append(n_frms)
    elif all([ os.path.exists("%s/Q.dat" % dirs[i]) for i in range(n_traj) ]):
        for i in range(n_traj):
            n_frms = file_len("%s/Q.dat" % dirs[i])
            traj_lengths.append(n_frms)
    return traj_lengths

def get_contact_function(function_type,args):
    """Wrap contact function"""
    if function_type not in supported_functions.keys():
        raise IOError("--function_type must be in " + supported_functions.__str__())
    else:
        fun = supported_functions[function_type]
        def contact_function(r):
            return fun(r,*args)
    return contact_function

def get_1D_contact_observable(pairs,function_type,contact_params,periodic):
    """Returns a function that takes a MDTraj Trajectory object"""
    if function_type not in supported_functions.keys():
        raise IOError("--function_type must be in " + supported_functions.__str__())
    else:
        contact_function = supported_functions[function_type]
        def obs_function(trajchunk):
            r = md.compute_distances(trajchunk,pairs,periodic=periodic)
            return np.sum(contact_function(r,*contact_params),axis=1)
    return obs_function

def get_2D_contact_observable(pairs,function_type,contact_params,periodic):
    """Returns a function that takes a MDTraj Trajectory object"""
    if function_type not in supported_functions.keys():
        raise IOError("--function_type must be in " + supported_functions.__str__())
    else:
        contact_function = supported_functions[function_type]
        def obs_function(trajchunk):
            r = md.compute_distances(trajchunk,pairs,periodic=periodic)
            return contact_function(r,*contact_params)
    return obs_function

def get_edwards_anderson_observable(pairs,function_type,contact_params,periodic):
    """ TODO: TEST. Returns a function that takes two MDTraj Trajectory objects"""
    if function_type not in supported_functions.keys():
        raise IOError("--function_type must be in " + supported_functions.__str__())
    else:
        contact_function = supported_functions[function_type]
        def obs_function(trajchunk1,trajchunk2):
            r1 = md.compute_distances(trajchunk1,pairs,periodic=periodic)
            r2 = md.compute_distances(trajchunk2,pairs,periodic=periodic)
            cont1 = contact_function(r1,*contact_params),axis=1)
            cont2 = contact_function(r2,*contact_params),axis=1)
            #np.dot(cont1,cont2)
            #return np.sum(contact_function(r,*contact_params),axis=1)
    return obs_function

######################################################################
# Contact functions
######################################################################
def tanh_contact(r,r0,widths):
    """Smoothly increasing tanh contact function"""
    return 0.5*(np.tanh(2.*(r0 - r)/widths) + 1)

def weighted_tanh_contact(r,r0,widths,weights):
    """Weighted smoothly increasing tanh contact function"""
    return weights*tanh_contact(r,r0,widths)

def step_contact(r,r0):
    """Step function indicator contact function"""
    return (r <= r0).astype(int)

def calculate_observable(trajfiles,observable_fun,topology,chunksize):
    """Calculate contacts by dynamically extending a list"""
    dirs = [ os.path.dirname(x) for x in trajfiles ] 
    n_traj = len(trajfiles) 
    observable = []
    for n in range(n_traj):
        logger.info("trajectory %s" % trajfiles[n])
        obs_traj = []
        # In order to save memory we loop over trajectories in chunks.
        for trajchunk in md.iterload(trajfiles[n],top=topology,chunk=chunksize):
            # Calculate observable for trajectory chunk
            obs_traj.extend(observable_fun(trajchunk))
        obs_traj = np.array(obs_traj)
        observable.append(obs_traj)
    return observable

@profile
def main():
    global supported_functions
    supported_functions = {"step":step_contact,"tanh":tanh_contact,"w_tanh":weighted_tanh_contact}

    # TODO: Test observable wrapper function

    source = "/home/ajk8/scratch/6-18-15_randnat/random_native_0.0001/replica_1/PDZ/iteration_0"

    # Just need to know:
    #   1. Where the data is: directories, trajectory names.
    #   2. Options for calculating observables: pairs,contact function parameters, mdtraj parameters

    # Data source
    dirs_file = "ticatemps"
    trajfiles = [ "%s/traj.xtc" % x.rstrip("\n") for x in open(dirs_file,"r").readlines() ]
    n_traj = len(trajfiles)

    dir0 = os.path.dirname(trajfiles[0])
    topology = "%s/Native.pdb" % dir0
    periodic = False

    chunksize = 1000 # Larger chunk size takes less time but more memory

    # Parameterization of contact-based reaction coordinate
    #contact_type = "Gaussian"
    function_type = "tanh"

    save_coord_as = "Qtanh.dat"
    native_pairs = np.loadtxt("%s/native_contacts.ndx" % dir0,skiprows=1,dtype=int) - 1
    n_native_pairs = len(native_pairs)
    r0_native = np.loadtxt("%s/pairwise_params" % dir0,usecols=(4,),skiprows=1)[1:2*n_native_pairs:2]
    widths = 0.1*np.ones(r0_native.shape[0],float)
    contact_params = (r0_native,widths)
    pairs = native_pairs

    # Setup logging to file and console
    logging.basicConfig(filename="contacts.log",
                        filemode="w",
                        format="%(levelname)s:%(asctime)s: %(message)s",
                        datefmt="%H:%M:%S",
                        level=logging.DEBUG)
    formatter = logging.Formatter("%(levelname)s:%(asctime)s: %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    global logger
    logger = logging.getLogger('contacts')
    logger.addHandler(console)

    # Parameterize contact function 
    #contact_function = get_contact_function(function_type,contact_params)
    Qtanh_function = get_1D_contact_observable(pairs,function_type,contact_params,periodic)
    Qitanh_function = get_2D_contact_observable(pairs,function_type,contact_params,periodic)

    logger.info("Calculating contacts for:")
    logger.info("%s" % trajfiles.__str__())
    starttime = time.time()
    contacts = calculate_observable(trajfiles,Qtanh_function,topology,chunksize)
    traj_lengths = [ len(contacts[i]) for i in range(n_traj) ]
    dt = time.time() - starttime
    logger.info("computation took %e sec , %e min"  % (dt,dt/60.))

    contacts = calculate_observable_fluctuations(trajfiles,Qitanh_function,contacts,topology,chunksize)

if __name__ == "__main__":
    main()
