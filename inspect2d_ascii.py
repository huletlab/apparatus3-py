#!/usr/bin/python

import sys
from numpy import loadtxt
import falsecolor
from matplotlib import cm


if __name__ == "__main__":
   # This program should be called as 
   # inspect2d_ascii.py  data.ascii fit.ascii inspec_row inspec_col  prefix
   data = loadtxt(sys.argv[1])
   fit  = loadtxt(sys.argv[2])

   row = sys.argv[3]
   col = sys.argv[4]

   prefix = sys.argv[5]

   #colormap = falsecolor.my_rainbow
   colormap = cm.spectral
   

   falsecolor.inspecpng( [data, fit], row, col, data.min(), data.max(), \
                         colormap, prefix, 100, origin = 'lower' ) 
  
   
   
 
