#!/bin/sh

if [ "$2" == "" ]; then
    echo " Second argument should be nsteps!"
else
    dirs=`cat $1`
    for x in ${dirs}; do
        cd ${x}
        steps=`grep "Statistics over" md.log | awk '{print $3}'`
        if [ "$steps" == "$2" ]; then
            echo " ${x} done "
        else
            echo $steps does not match $2, restarting ${x}
            qsub rst.pbs
        fi
        cd ../
    done
fi
