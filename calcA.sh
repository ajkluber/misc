#!/bin/bash
# Script to calculate total number of non-native contacts

dirs=`cat $1`
for x in ${dirs}; do
    cd ${x}
    # Grab non-native contact pair indices given pairwise_params file
    temp=`wc -l native_contacts.ndx | cut -c 1-3`
    n_nat=$(expr $temp - 1)
    n_nonnat=`cat pairwise_params | sed "1d" | awk '{print $1, $2, 0.15}' | uniq | sed "1,${n_nat}d" | wc -l`
    echo "${n_nonnat}" > nonnative_contacts.dat
    cat pairwise_params | sed "1d" | awk '{print $1, $2, 0.5}' | uniq | sed "1,${n_nat}d" >> nonnative_contacts.dat

    # Calculate total number of non-native contacts
    g_kuh_sbm -s Native.pdb -f traj.xtc -nc nonnative_contacts.dat -o A -noshortcut -abscut -cut 0.15
    mv A.out A.dat
    cd ../
done
