#!/bin/sh

dirs=`cat $1`
for x in ${dirs}; do
    cd ${x}
    echo "${x}" 
    echo "0" | g_gyrate_sbm -f traj.xtc -s topol_4.5.tpr -xvg none &> rg.dat
    awk '{print $2}' gyrate.xvg > Rg.dat
    rm gyrate.xvg
    cd ../
done
