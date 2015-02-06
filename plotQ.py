import argparse
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('--data', type=str, default="Q.dat", help='Name of timeseries to analyze.')
    parser.add_argument('--title', type=str, default="", help='Optional. Title for plot.')
    parser.add_argument('--saveas', type=str, default="None", help='Optional. Filename to save plot.')
    args = parser.parse_args()

    filename = args.data 
    coord = filename.split(".")[0]

    Q = np.loadtxt(filename)
    plt.plot(Q)
    plt.xlabel("frames",fontsize=20)
    plt.ylabel("%s" % coord,fontsize=20)
    plt.title(args.title)
    if args.saveas != "None":
        plt.savefig(args.saveas)
    plt.show()
