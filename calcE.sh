#!/bin/sh

dirs=`cat $2`


if [ $1 == cacb ]; then
    for x in ${dirs}; do
        cd ${x}
        pwd
        g_energy -f ener.edr -o energyterms -xvg none << EOF
Bond
Angle
Proper-Dih.
Improper-Dih.
Potential
EOF
        cd ../
    done
else
    for x in ${dirs}; do
        cd ${x}
        pwd
        g_energy -f ener.edr -o energyterms -xvg none << EOF
Bond
Angle
Coulomb-(SR)
Proper-Dih.
Potential
EOF
        cd ../
    done
fi
