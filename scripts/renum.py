import numpy as np
import shutil
import os

def renumber_contacts():
    permutants = [13,33,54,68,81]

    contacts = np.loadtxt("S6.contacts",dtype=int)

    for p in permutants:
        print "cp%d" % p
        newcontacts = np.ones(contacts.shape,int)

        idx = (contacts[:,1] <= p)
        newcontacts[idx,1] = contacts[idx,1] + 95 - p

        idx = (contacts[:,3] <= p)
        newcontacts[idx,3] = contacts[idx,3] + 95 - p

        idx = (contacts[:,1] > p)
        newcontacts[idx,1] = contacts[idx,1] - p

        idx = (contacts[:,3] > p)
        newcontacts[idx,3] = contacts[idx,3] - p


        for i in range(len(newcontacts)):
            cont = list(newcontacts[i,:])
            if cont[1] > cont[3]:
                newcontacts[i,1] = cont[3]
                newcontacts[i,3] = cont[1]
            if abs(cont[3] - cont[1]) < 4:
                print np.array([contacts[i,1],contacts[i,3]])
                #print cont

        np.savetxt("newcp%d.contacts" % p,newcontacts,fmt="%3d") 

def renumber_ddG():
    permutants = [13,33,54,68,81]

    lines = open("S6.ddG","r").readlines()

    for p in permutants:

        newddG = lines[0]
        for line in lines[1:]:
            residx = int(line[:7])
            if residx <= p:
                newidx = residx + 95 - p
            else:
                newidx = residx - p
            newline = "%6d%s" % (newidx,line[6:])
            newddG += newline

        open("newcp%d.ddG" % p,"w").write(newddG)

def convert_index(residx,p):
    if residx <= p:
       newidx = residx + 95 - p
    else:
       newidx = residx - p
    return newidx

def renumber_Fij():
    permutants = [13,33,54,68,81]
    #permutants = [13]

    lines = open("S6.ddG","r").readlines()

    for p in permutants:
        dir = "cp%d" % p
        if not os.path.exists("%s/mutants" % dir):
            os.mkdir("%s/mutants" % dir)
        if not os.path.exists("%s/mutants/wt.pdb" % dir):
            shutil.copy("%s/clean_noH.pdb" % dir, "%s/mutants/wt.pdb" % dir)
        if not os.path.exists("%s/mutants/core.ddG" % dir):
            shutil.copy("%s.ddG" % dir, "%s/mutants/core.ddG" % dir)
        if not os.path.exists("%s/mutants/contacts" % dir):
            shutil.copy("%s/contacts.dat" % dir,"%s/mutants/contacts" % dir)

        for line in lines[1:]:
            residx = int(line[:7])
            newidx = convert_index(residx,p)
            oldmut = "%s%d%s" % (line.split()[1],residx,line.split()[2])
            newmut = "%s%d%s" % (line.split()[1],newidx,line.split()[2])

            if not os.path.exists("%s/mutants/fij_%s.dat" % (dir,newmut)):
                oldfij = np.loadtxt("S6/mutants/fij_%s.dat" % oldmut)

                newfij = np.zeros(oldfij.shape)
                for n in range(len(oldfij)):
                    for m in range(len(oldfij)):
                        value = oldfij[n,m]
                        if value != 0:
                            new_n = convert_index(n+1,p) - 1
                            new_m = convert_index(m+1,p) - 1
                            if new_m < new_n:
                                newfij[new_m,new_n] = value
                            else:
                                newfij[new_n,new_m] = value
                np.savetxt("%s/mutants/fij_%s.dat"  % (dir,newmut),newfij,fmt="%.5f")
                print "saving %s/mutants/fij_%s.dat"  % (dir,newmut)
            

def renumber_secondary_structure():
    permutants = [13,33,54,68,81]

    lines = open("S6/secondary_structure.txt","r").readlines()

    for p in permutants:
        newlines = ""
        for line in lines:
            res1 = int(line.split()[1])
            res2 = int(line.split()[2])
            newres1 = convert_index(res1,p)
            newres2 = convert_index(res2,p)

            newline = "%s%3d%3d\n" % (line.split()[0],newres1,newres2)
            newlines += newline

        open("cp%d/secondary_structure.txt" % p,"w").write(newlines)

renumber_secondary_structure()
