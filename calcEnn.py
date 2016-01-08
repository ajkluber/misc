from simulation.calc.scripts.Enn import calc_Enn_for_directories

if __name__ == "__main__":
    size = 12
    trajfiles = [ "rank_{}/all_frames.xtc".format(i) for i in range(size) ]
    #trajfiles = [ "rank_0/all_frames.xtc" ]
    path_to_params = "/home/ajk8/scratch/6-10-15_nonnative/1E0G/random_b2_0.01/replica_1/1E0G/iteration_0/127.80_2"
    calc_Enn_for_directories(trajfiles, path_to_params=path_to_params)
