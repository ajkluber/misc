import argparse
import numpy as np

import model_builder as mdb


def draw_bond(atom1,atom2):
    molid = 0
    tclstring = ''
    tclstring +='set sel [atomselect %d "index %d %d"]\n' % (molid,atom1,atom2)
    tclstring += 'lassign [$sel getbonds] bond1 bond2\n'
    tclstring += 'set id [lsearch -exact $bond1 %d]\n' % atom2
    tclstring += 'if { $id == -1 } {\n'
    tclstring += 'lappend bond1 %d\n' % atom2
    tclstring += '}\n'
    tclstring += 'set id [lsearch -exact $bond2 %d]\n' % atom1
    tclstring += 'if { $id == -1 } {\n'
    tclstring += 'lappend bond2 %d\n' % atom1
    tclstring += '}\n'
    tclstring += '$sel setbonds [list $bond1 $bond2]\n'
    tclstring += '$sel delete\n\n'
    return tclstring


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('--pdb', type=str, default="Native.pdb", help='Name pdb.')
    parser.add_argument('--saveas', type=str, default="bonds.tcl", help='Name of .')
    args = parser.parse_args()

    pdb = args.pdb
    saveas = args.saveas 

    pdb = open(pdb,"r").read()

    pdb_info = mdb.models.pdb_parser.get_coords_atoms_residues(pdb)

    atm_coords = pdb_info[0]
    atm_indxs = pdb_info[1]
    atm_types = pdb_info[2]
    res_indxs = pdb_info[3]
    res_types = pdb_info[4]
    res_types_unique = pdb_info[5]

    CA_indxs = atm_indxs[atm_types == "CA"]
    CB_indxs = atm_indxs[atm_types == "CB"]

    n_residues = len(np.unique(np.array(res_indxs,copy=True)))

    tclstring = ''
    # Loop over CA's
    for i in range(n_residues-1):
        # First bond all the c-alphas together.
        tclstring += draw_bond(CA_indxs[i]-1,CA_indxs[i+1]-1)
        
    sub = 0
    # Loop over CA-CB pairs.
    for i in range(n_residues):
        # Then bond all c-alphas to their c-beta.
        if res_types_unique[i] == "GLY":
            # Skip glycine
            sub += 1
        else:
            tclstring += draw_bond(CA_indxs[i]-1,CB_indxs[i-sub]-1)

    open(saveas,"w").write(tclstring)
