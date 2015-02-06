#!/bin/sh

dirs=`cat $2`
if [ $1 == gauss ]; then
    for x in ${dirs}; do
        cd ${x}
        python -m misc.update_conts
        g_kuh_sbm -s conf.gro -f traj.xtc -n native_contacts.ndx -o Q -noshortcut -abscut -cut 0.1
        mv Q.out Q.dat
        cd ../
    done
else
    for x in ${dirs}; do
        cd ${x}
        python -m misc.update_conts
        g_kuh_sbm -s conf.gro -f traj.xtc -n native_contacts.ndx -o Q -noshortcut -noabscut -cut 0.2
        mv Q.out Q.dat
        cd ../
    done

fi
