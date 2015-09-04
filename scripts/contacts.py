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
    n_dirs = len(dirs)
    traj_lengths = [] 
    if all([ os.path.exists("%s/n_frames" % dirs[i]) for i in range(n_dirs) ]):
        for i in range(n_dirs):
            with open("%s/n_frames" % dirs[i],"r") as fin:
                n_frms = int(fin.read().rstrip("\n"))
            traj_lengths.append(n_frms)
    elif all([ os.path.exists("%s/Q.dat" % dirs[i]) for i in range(n_dirs) ]):
        for i in range(n_dirs):
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

@profile
def contacts_preallocated(dirs,trajfiles,traj_lengths,pairs,contact_function,topology,chunk,periodic):
    """Calculate contacts using preallocated array"""
    n_dirs = len(dirs) 
    contacts = [ np.empty(traj_lengths[i],float) for i in range(n_dirs) ]
    for n in range(n_dirs):
        logger.info("trajectory %s" % trajfiles[n])
        chunk_sum = 0
        # In order to save memory we loop over trajectories in chunks.
        for trajchunk in md.iterload(trajfiles[n],top=topology,chunk=chunk):
            chunk_len = trajchunk.n_frames

            r = md.compute_distances(trajchunk,pairs,periodic=periodic)
            cont_temp = np.sum(contact_function(r),axis=1)
            contacts[n][chunk_sum:chunk_sum + chunk_len] = cont_temp

            chunk_sum += chunk_len
        if not os.path.exists("%s/n_frames" % dirs[n]):
            with open("%s/n_frames" % dirs[n],"w") as fout:
                fout.write("%d" % contacts[n].shape[0])
        np.savetxt("%s/%s" % (dirs[n],save_coord_as),contacts[n])
    return contacts

@profile
def contacts_not_preallocated(dirs,trajfiles,pairs,contact_function,topology,chunk,periodic):
    """Calculate contacts by dynamically extending a list"""
    n_dirs = len(dirs) 
    contacts = [ [] for i in range(n_dirs) ]
    for n in range(n_dirs):
        logger.info("trajectory %s" % trajfiles[n])
        contacts_traj = []
        # In order to save memory we loop over trajectories in chunks.
        for trajchunk in md.iterload(trajfiles[n],top=topology,chunk=chunk):
            r = md.compute_distances(trajchunk,pairs,periodic=periodic)
            cont_temp = np.sum(contact_function(r),axis=1)
            contacts_traj.extend(cont_temp)

        contacts_traj = np.array(contacts_traj)
        contacts.append(contacts_traj)
        np.savetxt("%s/%s" % (dirs[n],save_coord_as),contacts_traj)

        traj_len = contacts_traj.shape[0]
        with open("%s/n_frames" % dirs[n],"w") as fout:
            fout.write("%d" % traj_len)

    return contacts

if __name__ == "__main__":
    global supported_functions
    supported_functions = {"step":step_contact,"tanh":tanh_contact,"w_tanh":weighted_tanh_contact}

    # Just need to know:
    #   1. Where the data is. directories, trajectory names format.
    #   2. Options for calculating observables

    # Data source
    temps_file = "ticatemps"
    dirs = [ x.rstrip("\n") for x in open(temps_file,"r").readlines() ]
    n_dirs = len(dirs)
    trajfiles = [ "%s/traj.xtc" % x for x in dirs ]
    topology = "%s/Native.pdb" % dirs[0]
    periodic = False

    chunk = 100 # Larger chunk size takes less time but more memory

    # Parameterization of contact-based reaction coordinate
    #contact_type = "Gaussian"
    contact_function = "tanh"
    save_coord_as = "Qtanh.dat"
    native_pairs = np.loadtxt("%s/native_contacts.ndx" % dirs[0],skiprows=1,dtype=int) - 1
    n_native_pairs = len(native_pairs)
    r0_native = np.loadtxt("%s/pairwise_params" % dirs[0],usecols=(4,),skiprows=1)[1:2*n_native_pairs:2]
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
    logger = logging.getLogger('contacts')
    logger.addHandler(console)

    # Parameterize contact function 
    contact_function = get_contact_function(contact_function,contact_params)

    traj_lengths = check_traj_lengths(dirs)
    traj_lengths = []

    logger.info("Calculating contacts for:")
    logger.info("%s" % trajfiles.__str__())
    starttime = time.time()
    if traj_lengths == []:
        contacts = contacts_not_preallocated(dirs,trajfiles,pairs,contact_function,topology,chunk,periodic)
        traj_lengths = [ len(contacts[i]) for i in range(n_dirs) ]
    else:
        contacts = contacts_preallocated(dirs,trajfiles,traj_lengths,pairs,contact_function,topology,chunk,periodic)
    dt = time.time() - starttime
    logger.info("computation took %e sec , %e min"  % (dt,dt/60.))


