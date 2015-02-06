#!/bin/sh


dirs=`cat $1`

for x in ${dirs}; do
    cd ${x}
    steps=`grep "Statistics over" md.log | awk '{print $3}'`
    echo $steps
    if [ $steps == 100000001 ]; then
        echo " "
    else
        qsub rst.pbs
    fi
    cd ../
done
