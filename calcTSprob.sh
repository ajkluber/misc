#!/bin/sh

dirs=`cat $1`
echo "running g_kuh"
for x in ${dirs}; do
    cd ${x}
    g_kuh_sbm -s Native.pdb -f traj.xtc -n native_contacts.ndx -noshortcut -abscut -cut 0.1 -qiformat list &> g_kuh.log
    mv qimap.out qimap.dat
    cd ../
done

echo "running TSprob"
python -m misc.TSprob $1

if [ "$2" == "" ]; then
    echo "removing qimap.dat"
    for x in ${dirs}; do
        cd ${x}
        rm qimap.dat
        cd ../
    done
fi
