#!/usr/bin/python

import numpy as np
import sys

if (len(sys.argv) < 2):
   print "Usage: ./group_kmeans.py kmeans*"
   sys.exit()

f = open("group_kmeans_output.txt", 'w')

for i in range(len(sys.argv) - 1):
   dat = np.loadtxt(sys.argv[i+1])
   sm = dat.sum(axis=0)
   out_str = str(sys.argv[i+1]) + "\n" + str(sm) + "\n"
   f.write(out_str)
   
    
	 
