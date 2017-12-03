#!/bin/bash

for i in `ls -d */`; do
	cd $i
    dir=`echo ${i%/}`
	for j in `ls *.au`; do
		
		realpath $j >> $dir.mf

	done
    sed -i "s/$/	$i" $dir.mf
	cd ../
	cat $dir/$dir.mf >> all.mf
    echo "finished $dir"
done


