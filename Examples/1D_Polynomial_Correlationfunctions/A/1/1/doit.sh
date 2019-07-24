#!/bin/bash

xpacdt.py -i sampling.in

for i in `ls -d ensm/trj_*`; do
#    j=`sed -e 's/[\/&]/\\&/g' 
    sed -e "s;TODO_FOLDER;${i};g" prop.in > prop_tmp.in
    xpacdt.py -i prop_tmp.in
    rm prop_tmp.in
done

xpacdt.py -i ana.in
