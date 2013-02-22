#!/usr/bin/python

import sys
import argparse
import os
import glob
from configobj import ConfigObj

import numpy

sys.path.append('/lab/software/apparatus3/py')

import falsecolor
import gaussfitter
import qrange


def rebin(a, shape):
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)

# --------------------- MAIN CODE  --------------------#

if __name__ == "__main__":
  parser = argparse.ArgumentParser('averageBragg.py')

  parser.add_argument('RANGE', action="store", type=str, help='average data in the specified range')
  parser.add_argument('KEY', action="store", type=str, help='average using this key')
  parser.add_argument('BINSZ', action="store", type=int, help='bin size')

  args = parser.parse_args()


  sec = args.KEY.split(':')[0]
  key = args.KEY.split(':')[1]
  

  i = 0
  keydict = {}
  sums = {}
  ns   = {}
  vals = {}

  shots = qrange.parse_range(args.RANGE)
  shlist = glob.glob( os.getcwd() + '/????_bragg.dat')
  shlist.sort()

  rangestr = args.RANGE.replace(':','to')
  rangestr = rangestr.replace(',','_')

  for sh in shlist: 
    for s in shots:
      if s in sh:
 
        report = ConfigObj( 'report'+s+'.INI')
        val = report[sec][key]
        print '#%s identified as %s:%s = %s' % (s,sec,key,val)
        if val in keydict.keys():
          index = keydict[val]

          #sums[index] = sums[index] + numpy.loadtxt( s + '_bragg.dat')
 
          dat = numpy.loadtxt( s + '_bragg.dat')          
          newshape = (dat.shape[0]/ args.BINSZ , dat.shape[1]/ args.BINSZ)
          redat = rebin(dat, newshape)

          sums[index] = sums[index] + redat

          ns[index] = ns[index] + 1 
        else:
          keydict[val] = i
          index =  i 
          i = i + 1

          #sums[index] = numpy.loadtxt( s + '_bragg.dat') 
 
          dat = numpy.loadtxt( s + '_bragg.dat')           
          newshape = (dat.shape[0]/ args.BINSZ , dat.shape[1]/ args.BINSZ)
          redat = rebin(dat, newshape)

          sums[index] = redat

          ns[index] = 1
 
        vals[index] = val
 
  for index in range(i):
    sums[index] = sums[index] / ns[index]
    pngprefix = 'braggAverage_' + rangestr +  '_' + key + '_' + vals[index] + 'rebin%d' % args.BINSZ
    print pngprefix
    extratext = ''
    col = sums[index].max(axis=0).argmax()
    row = sums[index].max(axis=1).argmax()
    falsecolor.inspecpng( [sums[index]], row, col, sums[index].min(), sums[index].max(), \
                         falsecolor.my_grayscale, pngprefix, 100, origin = 'upper' , step=True, scale=10, interpolation='nearest', extratext=extratext)
    
  print "Difference will be %s - %s" % (vals[1], vals[0])
  diff = sums[1] - sums[0] 
  col = diff.max(axis=0).argmax()
  row = diff.max(axis=1).argmax()
  col = 19./ args.BINSZ
  row = 19./ args.BINSZ
  pngprefix = "diffBragg_" + rangestr + 'rebin%d' % args.BINSZ
  print pngprefix
  extratext = ''
  falsecolor.inspecpng( [diff], row, col, diff.min(), diff.max(), \
                       falsecolor.my_grayscale, pngprefix, 100, origin = 'upper' , step=True, scale=10, interpolation='nearest', extratext=extratext)
  #print sums
