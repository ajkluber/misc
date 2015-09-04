import os
import sys
import argparse
import logging 
import numpy as np

import mdtraj as md

# Script utility for the calculation of contact based reaction coordinates


global supported_functions
supported_functions = {"step":step_contact,"tanh":tanh_contact}

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

######################################################################
# Contact function utilities
######################################################################
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

# Class SBM trajectories
#   - reaction coordinate
#   - pairs

def preallocate_if_possible(dirs):
    """Preallocated arrays for reaction coordinate"""
    n_dirs = len(dirs)
    if all([ os.path.exists("%s/n_frames" % dirs[i]) for i in range(n_dirs) ]):
        preallocated = True
        traj_n_frames = [] 
        for i in range(n_dirs):
            with open("%s/n_frames" % dirs[i],"r") as fin:
                n_frms = int(fin.read().rstrip("\n"))
            traj_n_frames.append(n_frms)
        n_frames = np.sum(traj_n_frames) 
        contacts = np.array(n_frames,float) 
    elif all([ os.path.exists("%s/Q.dat" % dirs[i]) for i in range(n_dirs) ]):
        preallocated = True
        traj_n_frames = [] 
        for i in range(n_dirs):
            n_frms = file_len("%s/Q.dat" % dirs[i])
            traj_n_frames.append(n_frms)
        n_frames = np.sum(traj_n_frames) 
        contacts = np.array(n_frames,float) 
    else:
        preallocated = False
        contacts = []
        traj_n_frames = []
    return preallocated,contacts,traj_n_frames

if __name__ == "__main__":

    # Need to specify

    # Data source
    temps_file = "ticatemps"
    dirs = [ x.rstrip("\n") for x in open(temps_file,"r").readlines() ]
    n_dirs = len(dirs)
    trajfiles = [ "%s/traj.xtc" % x for x in dirs ]
    topology = "%s/Native.pdb" % dirs[0]
    periodic = False

    # Parameterization of contact-based reaction coordinate
    contact_type = "Gaussian"
    save_coord_as = "Qtanh.dat"
    native_pairs = np.loadtxt("%s/native_contacts.ndx" % dirs[0],skiprows=1,dtype=int) - 1
    n_native_pairs = len(native_pairs)
    r0_native = np.loadtxt("%s/pairwise_params" % dirs[0],usecols=(4,),skiprows=1)[1:2*n_native_pairs:2]
    contact_params = (r0_native) 
    pairs = native_pairs

    # Parameterize contact function 
    contact_function = get_contact_function(contact_type,contact_params)

    #n_frames = np.sum([ file_len("%s/Q.dat" % dirs[i]) for i in range(len(dirs)) ])
    #Qtanh = np.zeros(n_frames,float)

    # Preallocate arrays if we can tell the length of trajectories
    preallocated,contacts,traj_n_frames = preallocate_if_possible(dirs)

    if preallocated:
        chunk_sum = 0
        # In order to save memory we loop over trajectories in chunks.
        for n in range(len(trajfiles)):
            traj_len = 0
            traj_start = chunk_sum
            for chunk in md.iterload(trajfiles[n],top=topology):
                chunk_len = chunk.n_frames

                r = md.compute_distances(chunk,pairs,periodic=periodic)
                cont_temp = np.sum(contact_function(r),axis=1)
                contacts[chunk_sum:chunk_sum + chunk_len,:] = cont_temp

                chunk_sum += chunk_len
                traj_len += chunk_len

            for i in range(n_dirs):
                np.savetxt("%s/%s" % (dirs[i],save_coord_as),contacts[traj_start:traj_start + traj_n_frames[n]])
        n_frames = np.sum(traj_n_frames)
    else:
        chunk_sum = 0
        # In order to save memory we loop over trajectories in chunks.
        for n in range(len(trajfiles)):
            traj_len = 0

            contacts_traj =  
            for chunk in md.iterload(trajfiles[n],top=topology):
                chunk_len = chunk.n_frames

                r = md.compute_distances(chunk,pairs,periodic=periodic)
                cont_temp = np.sum(contact_function(r),axis=1)
                if not preallocated:
                    contacts_traj.extend(cont_temp)
                else:
                    contacts[chunk_sum:chunk_sum + chunk_len,:] = cont_temp

                chunk_sum += chunk_len
                traj_len += chunk_len

            if not preallocated:
                contact.extend(contact_traj)
                traj_n_frames.append(traj_len)
                for i in range(n_dirs):
                    with open("%s/n_frames" % dirs[i],"w") as fout:
                        fout.write("%d" % traj_len)
                contact_traj = np.array(contact_traj)
            else:
                contact_traj = contacts[ :]

            for i in range(n_dirs):
                np.savetxt("%s/%s" % (dirs[i],save_coord_as),):
                    fout.write("%d" % traj_len)
        n_frames = np.sum(traj_n_frames)
        if not preallocated:
            contacts = np.array(contacts)

