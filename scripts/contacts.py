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
    """TODO: Test. Returns a function that takes a MDTraj Trajectory object"""
    if function_type not in supported_functions.keys():
        raise IOError("--function_type must be in " + supported_functions.__str__())
    else:
        contact_function = supported_functions[function_type]
        def obs_function(trajchunk):
            r = md.compute_distances(trajchunk,pairs,periodic=periodic)
            return contact_function(r,*contact_params)
    return obs_function

def get_edwards_anderson_observable(pairs,function_type,contact_params,periodic):
    """ NOT DONE. Returns a function that takes two MDTraj Trajectory objects"""
    if function_type not in supported_functions.keys():
        raise IOError("--function_type must be in " + supported_functions.__str__())
    else:
        contact_function = supported_functions[function_type]
        def obs_function(trajchunk1,trajchunk2):
            r1 = md.compute_distances(trajchunk1,pairs,periodic=periodic)
            r2 = md.compute_distances(trajchunk2,pairs,periodic=periodic)
            cont1 = contact_function(r1,*contact_params)
            cont2 = contact_function(r2,*contact_params)
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

def calculate_observable(trajfile,observable_fun,topology,chunksize):
    """Loop over chunks of a trajectory to calculate 1D observable"""
    # In order to save memory we loop over trajectories in chunks.
    obs_traj = []
    for trajchunk in md.iterload(trajfile,top=topology,chunk=chunksize):
        # Calculate observable for trajectory chunk
        obs_traj.extend(observable_fun(trajchunk))
    return np.array(obs_traj)

def calculate_observable_by_bins(trajfile,obs_by_bin,count_by_bin,observable_fun,binning_coord,bin_edges,topology,chunksize)
    """Loop over chunks of a trajectory to bin a set of observables along a 1D coordinate"""
    # In order to save memory we loop over trajectories in chunks.
    obs_traj = []
    start_idx = 0
    for trajchunk in md.iterload(trajfile,top=topology,chunk=chunksize):
        # Calculate observable for trajectory chunk
        obs_temp = observable_fun(trajchunk)
        chunk_size = trajchunk.n_frames
        coord = binning_coord[start_idx:start_idx + chunk_size]
        # Sort frames into bins along binning coordinate. Collect observable average
        for n in range(len(bin_edges) - 1):
            frames_in_this_bin = (coord >= bin_edges[n]) & (coord < bin_edges[n])
            if any(frames_in_this_bin):
                obs_by_bin[n,:] += np.sum(obs_temp[frames_in_this_bin,:],axis=0)
                count_by_bin[n,:] += float(sum(frames_in_this_bin))
        start_idx += chunk_size

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
    dirs = [ os.path.dirname(x) for x in trajfiles ] 
    n_traj = len(trajfiles)

    dir0 = dirs[0]
    topology = "%s/Native.pdb" % dir0
    periodic = False

    chunksize = 1000 # Larger chunk size takes less time but more memory

    # Parameterization of contact-based reaction coordinate
    #function_type = "tanh"
    function_type = "w_tanh"

    #save_coord_as = "Qtanh.dat"
    save_coord_as = "Qtanh_tica1.dat"
    native_pairs = np.loadtxt("%s/native_contacts.ndx" % dir0,skiprows=1,dtype=int) - 1
    n_native_pairs = len(native_pairs)
    r0_native = np.loadtxt("%s/pairwise_params" % dir0,usecols=(4,),skiprows=1)[1:2*n_native_pairs:2] + 0.1
    #widths = 0.1*np.ones(r0_native.shape[0],float)
    widths = 0.3*np.ones(r0_native.shape[0],float)
    contact_params = (r0_native,widths)
    pairs = native_pairs

    #tica_pairs = np.loadtxt("tica_nat_10/tica1_nat_10_1_weights.dat",usecols=(0,1),dtype=int)
    weights = np.loadtxt("tica_nat_10/tica1_nat_10_1_weights.dat",usecols=(2,))
    contact_params = (r0_native,widths,weights)

    # Setup logging to file and console
    logging.basicConfig(filename="contacts.log",
                        filemode="w",
                        format="%(levelname)s:%(asctime)s: %(message)s",
                        datefmt="%H:%M:%S",
                        level=logging.DEBUG)
    formatter = logging.Formatter("%(levelname)s:%(asctime)s: %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger = logging.getLogger('contacts')
    logger.addHandler(console)

    # Parameterize contact function 
    #Qtanh_function = get_1D_contact_observable(pairs,function_type,contact_params,periodic)
    Qtanh_function = get_1D_contact_observable(pairs,function_type,contact_params,periodic)

    logger.info("Calculating contacts for:")
    logger.info("%s" % trajfiles.__str__())
    starttime = time.time()
    traj_lengths = []
    contacts = []
    # calculate contacts
    for n in range(n_traj):
        logger.info("trajectory %s" % trajfiles[n])
        conts_traj = calculate_observable(trajfiles[n],Qtanh_function,topology,chunksize)
        contacts.append(conts_traj)
        if save_coord_as is not None:
            np.savetxt("%s/%s" % (dirs[n],save_coord_as),conts_traj)
    traj_lengths = [ len(contacts[i]) for i in range(n_traj) ]
    dt = time.time() - starttime
    logger.info("computation took %e sec , %e min"  % (dt,dt/60.))

    # Parameterize contact function
    Qitanh_function = get_2D_contact_observable(pairs,function_type,contact_params,periodic)

    n_bins = 30
    counts,bin_edges = np.histogram(concatenate(contacts),bins=n_bins)
    mid_bin = 0.5*(bin_edges[1:] + bin_edges[:-1])
    Qi_by_bin = np.zeros((n_bins,pairs.shape[0]),float)
    count_by_bin = np.zeros((n_bins,pairs.shape[0]),float)
    logger.info("binning Qtanh for each contact along global Qtanh")
    starttime = time.time()
    for n in range(n_traj):
        logger.info("trajectory %s" % trajfiles[n])
        calculate_observable_by_bins(trajfiles[n],Qi_by_bin,count_by_bin,Qitanh_function,contacts[n],bin_edges,topology,chunksize)
        contacts.append(conts_traj)
    dt = time.time() - starttime
    logger.info("computation took %e sec , %e min"  % (dt,dt/60.))

    avgQi_by_bin = (Qi_by_bin.T/count_by_bin).T



if __name__ == "__main__":
    main()
