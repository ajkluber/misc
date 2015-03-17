import argparse
import os 
import numpy as np

def get_contact_probability(state_label,state_bound):
    if not os.path.exists("cont_prob_%s.dat" % state_label):
        n_frames = 0.
        temps = [ x.rstrip("\n") for x in open("long_temps_last", "r").readlines() ]
        for i in range(len(temps)):
            T = temps[i]
            Q = np.loadtxt("%s/Q.dat" % T)
            qimap = np.loadtxt("%s/qimap.dat" % T)

            state_indicator = ((Q > state_bound[0]).astype(int)*(Q < state_bound[1]).astype(int)).astype(bool)
            n_frames += float(sum(state_indicator.astype(int)))
            if i == 0:
                contact_probability = sum(qimap[(state_indicator == True),:])
            else:
                contact_probability += sum(qimap[(state_indicator == True),:])
        contact_probability /= n_frames
        np.savetxt("cont_prob_%s.dat" % state_label,contact_probability)
    else:
        contact_probability = np.loadtxt("cont_prob_%s.dat" % state_label)
    return contact_probability

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('--name', type=str, required=True, help='Name of protein to plot.')
    parser.add_argument('--iteration', type=int, required=True, help='Iteration to plot.')
    args = parser.parse_args()

    name = args.name
    iteration = args.iteration

    os.chdir("%s/iteration_%d" % (name,iteration))

    state_labels = []
    state_bounds = []
    for line in open("state_bounds.txt","r"):
        state_labels.append(line.split()[0])
        state_bounds.append([int(line.split()[1]),int(line.split()[2])])

    for i in range(len(state_labels)): 
        get_contact_probability(state_labels[i],state_bounds[i])
    os.chdir("../..")
