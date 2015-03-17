import argparse
import numpy as np

def draw_contact(ndx1,ndx2,pairnum):
    tclstring = ''
    tclstring +='set sel%d [atomselect top "resid %d and name CA and chain A"]\n' % (ndx1,ndx1)
    tclstring +='set sel%d [atomselect top "resid %d and name CA and chain A"]\n' % (ndx2,ndx2)
    tclstring +='lassign [atomselect%d get {x y z}] pos1\n' % (pairnum*2)
    tclstring +='lassign [atomselect%d get {x y z}] pos2\n' % (pairnum*2 + 1)
    tclstring +='draw color green\n'
    tclstring +='draw line $pos1 $pos2 style solid width 2\n'
    return tclstring

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('--contacts', type=str, default="native_contacts.ndx", help='Name contact file.')
    parser.add_argument('--start', type=int, default=0,help='Name contact file.')
    args = parser.parse_args()

    pairs = np.loadtxt(args.contacts,dtype=int,skiprows=1)
    counter = args.start

    tclstring = ''
    # Loop over pairs
    for i in range(len(pairs)):
        tclstring += draw_contact(pairs[i,0],pairs[i,1],counter)
        counter += 1

    #tclstring += 'mol modselect 0 top "all"'

    open("conts.tcl","w").write(tclstring)
