import os
import pdb
import argparse
import numpy as np

from memory_profiler import profile

#import model_builder.models.pdb_parser as pdb_parser

import misc.scripts.util as util

def get_args():
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
    return args

if __name__ == "__main__":
    args = get_args()

    dirsfile = args.dirs
    function = args.function 
    chunksize = args.chunksize
    topology = args.topology
    periodic = args.periodic

    util.check_if_supported(function)

    if args.saveas is None:
        save_coord_as = {"step":"Q.dat","tanh":"Qtanh.dat","w_tanh":"Qtanh_w.dat"}[function]
    else:
        save_coord_as = args.saveas

    # Data source
    cwd = os.getcwd()
    trajfiles = [ "%s/%s/traj.xtc" % (cwd,x.rstrip("\n")) for x in open(dirsfile,"r").readlines() ]
    dir = os.path.dirname(trajfiles[0])
    n_native_pairs = len(open("%s/native_contacts.ndx" % dir).readlines()) - 1
    r0 = np.loadtxt("%s/pairwise_params" % dir,usecols=(4,),skiprows=1)[1:2*n_native_pairs:2] + 0.1

    # Get contact function parameters
    if function == "w_tanh":
        if (not os.path.exists(args.tanh_weights)) or (args.tanh_weights is None):
            raise IOError("Weights file doesn't exist: %s" % args.tanh_weights)
        else:
            pairs = np.loadtxt(args.tanh_weights,usecols=(0,1),dtype=int)
            widths = args.tanh_scale*np.ones(pairs.shape[0],float)
            weights = np.loadtxt(args.tanh_weights,usecols=(2,),dtype=float)
            contact_params = (r0,widths,weights)
    elif function == "tanh":
        pairs = np.loadtxt("%s/native_contacts.ndx" % dir,skiprows=1,dtype=int) - 1
        widths = args.tanh_scale*np.ones(r0.shape[0],float)
        contact_params = (r0,widths)
    elif function == "step":
        pairs = np.loadtxt("%s/native_contacts.ndx" % dir,skiprows=1,dtype=int) - 1
        contact_params = (r0)
    else:
        raise IOError("--function must be in: %s" % util.supported_functions.keys().__str__())

    # Parameterize contact-based reaction coordinate
    contact_function = util.get_sum_contact_function(pairs,function,contact_params,periodic=periodic)

    # Calculate contact function over directories
    util.calc_coordinate_multiple_trajs(trajfiles,contact_function,topology,chunksize,save_coord_as=save_coord_as)
