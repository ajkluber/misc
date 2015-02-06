import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

#from project_tools.analysis.plot.drawss import add_secondary_struct_icons

def plot_Qfoldons_map(name):
    """ Plot the ss Qgroups on a contact map and save Qgroups"""
    contacts = np.loadtxt("%s/contacts.dat" % name,dtype=int)

    n_groups = len(open("%s/Qfoldons/colors.txt" % name, "r").readlines())
    Qgrp_indxs = []
    for i in range(n_groups):
        temp = [ int(x.rstrip("\n")) for x in open("%s/Qfoldons/group%d.dat" % (name,i),"r").readlines() ]
        Qgrp_indxs.append(temp)

    colors = [ x.rstrip("\n") for x in open("%s/Qfoldons/colors.txt" % name, "r").readlines() ]
                    
    fig = plt.figure() 
    ax = fig.add_subplot(111) 
    i = 0               
    for n in range(n_groups):
        print "group %d plotted in color %s" % (n,colors[n])
        for m in range(len(Qgrp_indxs[n])):
            pair = contacts[Qgrp_indxs[n][m],:]
            ax.plot(pair[0]-1,pair[1]-1,marker='s',color=colors[n])
            i += 1

    N = int(max(contacts.ravel()))
    plt.xlim(0,N)
    plt.ylim(0,N)
    plt.xticks(range(0,N,10))
    plt.yticks(range(0,N,10))
    plt.grid(True)
    #plt.legend(loc=4)
    plt.title("Q$_{foldon}$ groups",fontsize=20)
    plt.savefig("%s/Qfoldons/legend.png" % name)
    plt.savefig("%s/Qfoldons/legend.pdf" % name)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('--name', type=str, required=True, help='Name of protein to plot.')
    args = parser.parse_args()

    name = args.name

    plot_Qfoldons_map(name)
