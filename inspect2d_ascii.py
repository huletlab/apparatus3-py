#!/lab/software/epd-7.0-1-rh5-x86/bin/python

import sys
from numpy import loadtxt
import falsecolor


if __name__ == "__main__":
   # This program should be called as 
   # inspect2d_ascii.py  data.ascii fit.ascii inspec_row inspec_col  prefix
   data = loadtxt(sys.argv[1])
   fit  = loadtxt(sys.argv[2])

   row = sys.argv[3]
   col = sys.argv[4]

   prefix = sys.argv[5]

   falsecolor.inspecpng( [data, fit], row, col, data.min(), data.max(), \
                         falsecolor.my_rainbow, prefix, 100, origin = 'upper' ) 
  
   
   
 
