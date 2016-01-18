import numpy as np
import argparse

from simulation.calc.scripts.Enn import calc_Enn_for_directories

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Energy minimization for inherent structure analysis.")
    parser.add_argument("--size",
                        type=int,
                        required=True,
                        help="Number of subdirs.")

    parser.add_argument("--path_to_params",
                        type=str,
                        required=True,
                        help="Path to params.")

    args = parser.parse_args()
    path_to_params = args.path_to_params
    size = args.size
    #path_to_params = "/home/ajk8/scratch/6-10-15_nonnative/1E0G/random_b2_0.01/replica_1/1E0G/iteration_0/127.80_2"
    
    trajfiles = [ "rank_{}/all_frames.xtc".format(i) for i in range(size) ]
    calc_Enn_for_directories(trajfiles, path_to_params=path_to_params)
