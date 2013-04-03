#!/usr/bin/python

import sys
import argparse
import os
import glob
from configobj import ConfigObj

import numpy as np
import matplotlib.pyplot as plt
import itertools

sys.path.append('/lab/software/apparatus3/py')

import falsecolor
import gaussfitter
import qrange
import statdat


# --------------------- MAIN CODE  --------------------#

if __name__ == "__main__":
  parser = argparse.ArgumentParser('qstat.py')

  parser.add_argument('range', action = "store", \
         help="range of shots to be used.")

  parser.add_argument('XKEY', action="store", \
         type=str, help='using this key as discriminator')

  parser.add_argument('YKEYS', action="store", nargs='*',\
         help='get statistics for all keys in this list')


  args = parser.parse_args()

  #print args.range
  #print args.XKEY
  #print args.YKEYS

  rangestr = args.range.replace(':','to')
  rangestr = rangestr.replace(',','_')
  shots = qrange.parse_range(args.range)

  data = []
  for s in shots:
      report = ConfigObj( 'report'+s+'.INI')
      line = [ qrange.evalstr( report, args.XKEY ) ] 
      for Y in args.YKEYS:
          line.append( qrange.evalstr( report, Y)  )
      data.append(line)
  
  dat = np.array(data)
  #print dat

  out = statdat.statdat( dat, 0, 1)[:,0]
 
  header = '#\n# Column index\n#  0  %s\n' % args.XKEY
  for i,Y in enumerate(args.YKEYS):
      header = header + '#  %d  %s\n' % ( 3*i+1, Y )
      header = header + '#  %d  %s\n' % ( 3*i+2, 'standard deviation' )
      header = header + '#  %d  %s\n' % ( 3*i+3, 'standard error of the mean' )
      header = header + '#  %d  %s\n' % ( 3*i+4, 'pk-pk' )
      try:
          out = np.c_[ out, statdat.statdat( dat, 0, i+1 )[:,1:] ]
      except:
          continue

  print header,

  format = '%8.3f'
  ncol = out.shape[1]
  format = [format ,] *ncol
  format = ' '.join(format)
      
  for row in out:
      print format % tuple(row)  

  exit(1)

 

  

