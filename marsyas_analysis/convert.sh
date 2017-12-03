#!/bin/bash

for i in `ls *`; do
	name=`echo ${i::-4}`
	wav=".wav"
	new_name=$name$wav
	ffmpeg -i $i $new_name
done
