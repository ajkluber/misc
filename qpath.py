
import os
import shutil
import argparse
import numpy as np

def get_early_late_contacts(name):
    os.chdir("%s/iteration_0" % name)
    if os.path.exists("early_conts"):
        early = np.loadtxt("early_conts",dtype=int)
        late = np.loadtxt("late_conts",dtype=int)
    else:
        TS = np.loadtxt("cont_prob_TS.dat")
        early = np.where(TS > 0.5)[0]
        late = np.where(TS <= 0.5)[0]
        np.savetxt("early_conts",early,fmt="%4d")
        np.savetxt("late_conts",late,fmt="%4d")
    os.chdir("../..")
    return early,late

def calculate_Qpath(early,late):

    temps = [ x.rstrip("\n") for x in open("long_temps_last","r").readlines() ]

    n_early = float(len(early))
    n_late = float(len(late))
    n_total = n_early + n_late

    for i in range(len(temps)):
        print "  calculating for %s" % temps[i]
        qimap = np.loadtxt("%s/qimap.dat" % temps[i],dtype=int)

        qearly = np.sum(qimap[:,early],axis=1)
        qlate = np.sum(qimap[:,late],axis=1)

        qpath = (qearly - qlate) - (n_early - n_late)*(qearly + qlate)/n_total

        np.savetxt("%s/qpath.dat" % temps[i],qpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('--name', type=str, required=True, help='Name of protein to plot.')
    parser.add_argument('--iteration', type=int, required=True, help='Iteration to plot.')
    args = parser.parse_args()

    name = args.name
    iteration = args.iteration

    early,late = get_early_late_contacts(name)
    os.chdir("%s/iteration_%d" % (name,iteration))
    calculate_Qpath(early,late)
    os.chdir("../..")
