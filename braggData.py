#!/usr/bin/python

import argparse
import sys 
import pyfits
import wx
import numpy
import glob

from configobj import ConfigObj

sys.path.append('/lab/software/apparatus3/bin/py')
import falsecolor
import gaussfitter
import qrange


def get_counts( out , box):
  #print out.max()
  col = out.max(axis=0).argmax()
  row = out.max(axis=1).argmax()
  print "Max at (%d,%d) = %d" % ( row,col, out[row,col])

  #print args.SIZE
  br0 = 1+row+numpy.ceil(-box/2)
  br1 = 1+row+numpy.ceil(box/2)
  bc0 = 1+col+numpy.ceil(-box/2)
  bc1 = 1+col+numpy.ceil(box/2)
 
  bragg = out[ br0:br1 , bc0:bc1]

  mask = numpy.ones( out.shape)
  mask[ br0:br1, bc0:bc1 ] = 0.
  offset = numpy.sum( mask * out ) / ( out.shape[0]*out.shape[1] - bragg.shape[0]*bragg.shape[1] ) 
 
  patch = numpy.ones( out.shape) * offset 

  braggcounts = numpy.sum(numpy.sum(bragg))
  patchcounts = bragg.shape[0]*bragg.shape[1]*offset
  return braggcounts, patchcounts, col, row, patch

def analyze_path( mantapath , pngprefix, roi, box):
  print "Analyzing %s" % e
  shotnum =  os.path.basename( mantapath ).split('atoms.manta')[0]
  shotnum = "%04d" % int(shotnum)

  atomsfile = shotnum + 'atoms.manta'


  shot = atomsfile.split('atoms')[0]
  atoms     = numpy.loadtxt( shot + 'atoms.manta')
  noatoms   = numpy.loadtxt( shot + 'noatoms.manta')
  atomsref  = numpy.loadtxt( shot + 'atomsref.manta')
  noatomsref= numpy.loadtxt( shot + 'noatomsref.manta')

  operation = 'PHC' 
  
  if operation == 'ABS':
    out = (atoms - atomsref) / (noatoms - noatomsref)
  elif operation == 'PHC':
    out = (atoms - atomsref) - (noatoms - noatomsref) 
  else:
    print " -->  Operation is not ABS or PHC.  Program will exit"
    exit(1) 
   
  if roi:
    X0 = float(roi.split(',')[0])
    Y0 = float(roi.split(',')[1])
    XW = float(roi.split(',')[2])
    YW = float(roi.split(',')[3])
    out = out[Y0:Y0+YW, X0:X0+XW]
  else:
    X0 = 0.
    Y0 = 0.
    XW = out.shape[0]
    YW = out.shape[1] 
  #print out.shape

  numpy.savetxt(shot+'_bragg.dat', out, fmt='%d')

  braggcounts, patchcounts, col, row, patch = get_counts(out, box) 
  braggsig = braggcounts - patchcounts

  pstart = [patchcounts/box/box,braggsig, row, col, 5., 5., 0.] 
  p, fitimg =  gaussfitter.gaussfit( out, params=pstart, returnfitimage=True)
  # p = [height, amplitude, x, y, width_x, width_y, rotation]
  braggsig_gaus = p[1]*numpy.absolute(p[4])*numpy.absolute(p[5])*2*numpy.pi


  extratext = "   Bragg(gauss) = %.0f" % braggsig_gaus

 
  pngprefix = pngprefix + shotnum + '_bragg'
  falsecolor.inspecpng( [out, fitimg, patch], row, col, out.min(), out.max(), \
                         falsecolor.my_grayscale, pngprefix, 100, origin = 'upper' , step=True, scale=10, interpolation='nearest', extratext=extratext)
  


  inifile = "report" + shotnum + ".INI"
  report = ConfigObj(inifile)
 
  report['BRAGG'] = {} 
  report['BRAGG']['rawbragg'] = braggcounts
  report['BRAGG']['patch'] = patchcounts
  report['BRAGG']['bragg'] = braggsig
  
  report.write()
 
  print "%s -->  Bragg(gauss) = %6.0f ,  Bragg = %6.0f ,  Raw =  %6d ,  Offset = %6.0f" % (shotnum, braggsig_gaus, braggsig, braggcounts, patchcounts) 
  


# --------------------- MAIN CODE  --------------------#


if __name__ == "__main__":
  parser = argparse.ArgumentParser('braggData.py')
  parser.add_argument('SIZE', action="store", type=int, help='size in pixels of bragg peak region')
  parser.add_argument('-r', action="store", dest='ROI', type=str, help='XO,Y0,XW,YW region of interest')
  parser.add_argument('-d', action = "store", dest='DIR', type=str, help="path of directory for output png file")
  parser.add_argument('--backwards', action = "store_true", dest='BACKWARDS', help="analyze files already on disk starting with the last one.  prompt at each")
  parser.add_argument('--forcebackwards', action = "store_true", dest='FBACKWARDS', help="analyze files already on disk starting with the last one.  DO NOT prompt at each")
  parser.add_argument('--range', action = "store", dest='range', help="analyze files in the specified range.  DO NOT prompt at each")
 
  args = parser.parse_args()
  #print type(args)
  #print args
 

  import monitordir
  import os

  if args.DIR:
     pngprefix = args.DIR  
  else:
     pngprefix = ''
 
  
  if args.range != None:
    shots = qrange.parse_range( args.range)
    list = glob.glob( os.getcwd() +  '/????atoms.manta' )
    list.sort()
    for e in list:
      for s in shots:
        if s in e:
          try:
            analyze_path( e, pngprefix, args.ROI, args.SIZE)
          except:
            print "ERROR analyzing %s" % e
    exit(0)
    

  if args.BACKWARDS or args.FBACKWARDS:
    list = glob.glob( os.getcwd() +  '/????atoms.manta' )
    list.sort()
    list.reverse()
    print list
    for e in list:
      try:
        analyze_path( e, pngprefix, args.ROI, args.SIZE)
      except:
        print "ERROR analyzing %s" % e
      if args.FBACKWARDS:
        cont = 'y'
      else:
        cont = raw_input("Continue backwards? [y/n]")
      if cont != 'y':
        exit(1)
    exit(0)
   
  list = glob.glob( os.getcwd() +  '/????atoms.manta' )
  list.sort()
  e=list[-1]
  analyze_path( e, pngprefix, args.ROI, args.SIZE)

  while (1):
    new = monitordir.waitforit( os.getcwd(), '/????atoms.manta' )
    for e in new:
      analyze_path( e, pngprefix, args.ROI, args.SIZE)
