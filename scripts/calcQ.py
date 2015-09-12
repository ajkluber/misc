import os
import argparse
import numpy as np

from memory_profiler import profile

import misc.scripts.util as util

def calculate_contacts(pairs,function,contact_params,trajfiles,topology,
        collect=False,save_coord_as=None,chunksize=1000,periodic=False):
    """ 
    Parameters
    ----------
    pairs : np.array (n_pairs,2)
        Pairs of resiudes to compute contacts between.

    function : str
        Name of contact function to compute.

    contact_params : tuple
        Parameters for corresponding contact function.

    trajfiles : list
        List of trajectory files.

    topology : str
        Filename of molecular topology (e.g. pdb file).
    
    collect : bool, opt.
        Return contact timeseries for each trajectory. 

    save_coord_as : str
        If given each contact timeseries is saved in corresponding subdirectory.
    
    chunksize : int
        Number of frames that mdtraj loads when iterating over trajectory.

    periodic : bool
        Passed to mdtraj. Compute pairwise distances using minimum image convention.
    
    """
    # Parameterization of contact-based reaction coordinate
    contact_function = util.get_1D_contact_observable(pairs,function,contact_params,periodic=periodic)

    # Calculate contacts for each trajectory
    contacts = [] 
    for n in range(n_traj):
        dir = os.path.trajfiles[n]
        logger.info("trajectory %s" % trajfiles[n])
        contacts_traj = calculate_observable(trajfiles[n],contact_function,topology,chunksize)
        if save_coord_as is not None:
            np.savetxt("%s/%s" % (dir,save_coord_as),contacts_traj)
        if collect: 
            contacts.append(contacts_traj)
    return contacts

@profile
def main(args):
    dirsfile = args.dirs
    function = args.function 
    chunksize = args.chunksize
    topology = args.topology
    periodic = args.periodic

    util.check_if_supported(function)

    if args.saveas is None:
        save_coord_as = {"step":"Q.dat","tanh":"Qtanh.dat","w_tanh":"Qtanh_w.dat"}[function]
    else:
        saveas_coord_as = args.saveas

    # Data source
    cwd = os.getcwd()
    trajfiles = [ "%s/%s/traj.xtc" % (cwd,x.rstrip("\n")) for x in open(dirsfile,"r").readlines() ]
    n_traj = len(trajfiles)
    dir = os.path.dirname(trajfiles[0])
    r0 = np.loadtxt("%s/pairwise_params" % dir,usecols=(4,),skiprows=1)[1:2*n_native_pairs:2] + 0.1

    if function == "w_tanh":
        if (not os.path.exists(args.tanh_weights)) or (args.tanh_weights is None):
            raise IOError("Weights file doesn't exist: %s" % args.tanh_weights)
        else:
            pairs = np.loadtxt(args.tanh_weights,usecols=(0,1),dtype=pairs)
            widths = args.tanh_scale*np.ones(r0.shape[0],float)
            weights = np.loadtxt(args.tanh_weights,usecols=(2,),dtype=float)
            contact_params = (r0,widths,weights)
    elif function == "tanh":
        pairs = np.loadtxt("%s/native_contacts.ndx" % dir,skiprows=1,dtype=int) - 1
        widths = args.tanh_scale*np.ones(r0.shape[0],float)
        contact_params = (r0,widths)
    elif function == "step":
        contact_params = (r0)
    else:
        raise IOError("--function must be in: %s" util.supported_functions.keys().__str__())

    calculate_contacts(pairs,function,contact_params,trajfiles,topology,
            collect=False,save_coord_as=save_coord_as,chunksize=chunksize,periodic=periodic)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('--dirs',
            type=str,
            required=True,
            help='File holding directory names.')

    parser.add_argument('--function',
            type=str,
            required=True,
            help='Contact functional form.')

    parser.add_argument('--topology',
            type=str,
            default="Native.pdb",
            help='Contact functional form. Opt.')

    parser.add_argument('--tanh_scale',
            default=0.3,
            help='Tanh contact switching scale. Opt.')

    parser.add_argument('--tanh_weights',
            type=float,
            help='Tanh contact weights. Opt.')

    parser.add_argument('--chunksize',
            type=int,
            default=1000,
            help='Chunk size to parse traj.')

    parser.add_argument('--periodic',
            type=bool,
            default=False,
            help='Periodic.')

    parser.add_argument('--saveas',
            type=str,
            help='File to save in directory.')

    args = parser.parse_args()

    main(args)

