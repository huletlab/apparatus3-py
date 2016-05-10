#!/usr/bin/python

import sys
from numpy import loadtxt
import falsecolor
from matplotlib import cm


if __name__ == "__main__":
   # This program should be called as 
   # inspect2d_ascii.py  data.ascii,fit1.ascii,fit2.ascii,fit3.ascii,... inspec_row inspec_col  prefix
   list = sys.argv[1].split(',')
   imgs =[]
   for file in list:
      imgs.append( loadtxt (file) )
     

   row = sys.argv[2]
   col = sys.argv[3]

   prefix = sys.argv[4]

   #colormap = falsecolor.my_rainbow
   colormap = cm.spectral

   falsecolor.inspecpng( imgs, row, col, imgs[0].min(), imgs[0].max(), \
                         colormap, prefix+'_multi', 100, origin = 'upper' ) 
  
   
   
 
