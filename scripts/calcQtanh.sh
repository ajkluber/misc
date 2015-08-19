#!/bin/sh

dirs=`cat $1`
for x in ${dirs}; do
    cd ${x}
    g_kuh_sbm -s Native.pdb -f traj.xtc -n native_contacts.ndx -o Qtanh -kappa 5
    mv Qtanh.out Qtanh.dat
    cd ../
done
