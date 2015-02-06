import numpy as np
import glob 
import os


allfiles = glob.glob("*/*/*/contacts.dat")

cwd = os.getcwd()
for i in range(len(allfiles)):
#for i in range(1):
    dir = allfiles[i].split("contacts.dat")[0]
    os.chdir(dir)

    conts = np.loadtxt("contacts.dat",dtype=int)
    #nat_conts = conts
    nat_conts = conts[::2]
    np.savetxt("native_contacts.dat",nat_conts,fmt="%5d")
    nat_cont_ndx = "[ native_contacts ]\n"
    for i in range(len(nat_conts)):
        nat_cont_ndx += "%5d%5d\n" % (nat_conts[i,0],nat_conts[i,1])
    open("native_contacts.ndx","w").write(nat_cont_ndx)

    os.chdir(cwd)
