import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

#from project_tools.analysis.plot.drawss import add_secondary_struct_icons

def plot_Qgroups_ss_map(name,skip):
    """ Plot the ss Qgroups on a contact map and save Qgroups"""
    conts = np.loadtxt("%s/contacts.dat" % name,dtype=int)
    contacts = conts[::skip,:]
    ## Read in secondary structure assignment.
    ss_labels = []
    ss_bounds = []
    for line in open("%s/secondary_structure.txt" % name,"r").readlines():
        ss_labels.append(line.split()[0])
        ss_bounds.append([int(line.split()[1]),int(line.split()[2])])
    n_ss_elements = len(ss_labels)

    n_groups = len(open("%s/Qgroups_ss/labels.txt" % name, "r").readlines())
    Qgrp_indxs = []
    for i in range(n_groups):
        temp = [ int(x.rstrip("\n")) for x in open("%s/Qgroups_ss/group%d.dat" % (name,i),"r").readlines() ]
        Qgrp_indxs.append(temp)

    colors = [ x.rstrip("\n") for x in open("%s/Qgroups_ss/colors.txt" % name, "r").readlines() ]
                    
    fig = plt.figure() 
    ax = fig.add_subplot(111) 
    i = 0               
    for n in range(n_groups):
        print "group %d plotted in color %s" % (n,colors[n])
        for m in range(len(Qgrp_indxs[n])):
            pair = contacts[Qgrp_indxs[n][m],:]
            ax.plot(pair[0]-1,pair[1]-1,marker='s',color=colors[n])
            i += 1

    ticks = []
    for a,b in ss_bounds:
        ticks.append(a)

    N = int(max(contacts.ravel()))
    #add_secondary_struct_icons(ax,ss_labels,ss_bounds)
    plt.xlim(0,N)
    plt.ylim(0,N)
    plt.xticks(ticks)
    plt.yticks(ticks)
    #plt.setp(ax.get_xticklabels(), visible=False)
    #plt.setp(ax.get_yticklabels(), visible=False)
    plt.grid(True)
    #plt.legend(loc=4)
    plt.savefig("%s/Qgroups_ss/legend.png" % name)
    plt.savefig("%s/Qgroups_ss/legend.pdf" % name)
    plt.title("Q$_{sec struct}$ groups")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('--name', type=str, required=True, help='Name of protein to plot.')
    parser.add_argument('--skip', type=int, default=1, help='Name of protein to plot.')
    args = parser.parse_args()

    name = args.name
    skip = args.skip

    plot_Qgroups_ss_map(name,skip)
