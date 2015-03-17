import argparse
import numpy as np


def get_exp_phi_res(name):

    n_residues = len(open("%s/Native.pdb" % name,"r").readlines()) - 1

    lines = [ x.rstrip("\n") for x in open("%s/mutants/core.ddG" % name,"r").readlines() ]

    exp_phi_res = np.zeros(n_residues,float)

    for line in lines[1:]:
        vals = line.split()
        if int(vals[8]) == 1:
            continue
        else:
            phi = float(vals[7])
            mut_idx = int(vals[0])
            if phi == -999.:
                exp_phi_res[mut_idx - 1] = 0
            else:
                exp_phi_res[mut_idx - 1] = phi
                print mut_idx, phi


    pdb = open("%s/clean.pdb" % name,"r").readlines()
    newpdb = ""
    for i in range(len(pdb)):
        line = pdb[i]
        if line.startswith("END"):
            newpdb += "END"
            break
        else:
            resnum = int(line[22:26]) - 1
            newpdb += "%s     %5f\n" % (line[:55],exp_phi_res[resnum])

    open("%s/exp_phi.pdb" % name,"w").write(newpdb)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('--name', type=str, required=True, help='Name of protein to plot.')
    args = parser.parse_args()

    name = args.name

    get_exp_phi_res(name)

