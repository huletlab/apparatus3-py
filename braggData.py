#!/lab/software/epd-7.0-1-rh5-x86/bin/python

import argparse
import sys 
import pyfits
import wx
import numpy
import glob

from configobj import ConfigObj

sys.path.append('/lab/software/apparatus3/bin/py')
import falsecolor


def makepng( atomsfile, operation, dpi, fileprefix, ROI):
  #file prefix can include a direcdtory where the png files will be located
  shot = atomsfile.split('atoms')[0]
  atoms     = numpy.loadtxt( shot + 'atoms.manta')
  noatoms   = numpy.loadtxt( shot + 'noatoms.manta')
  atomsref  = numpy.loadtxt( shot + 'atomsref.manta')
  noatomsref= numpy.loadtxt( shot + 'noatomsref.manta') 
  
  if operation == 'ABS':
    out = (atoms - atomsref) / (noatoms - noatomsref)
  elif operation == 'PHC':
    out = (atoms - atomsref) - (noatoms - noatomsref) 
  else:
    print " -->  Operation is not ABS or PHC.  Program will exit"
    exit(1) 
   
  if ROI:
    X0 = float(ROI.split(',')[0])
    Y0 = float(ROI.split(',')[1])
    XW = float(ROI.split(',')[2])
    YW = float(ROI.split(',')[3])
    out = out[Y0:Y0+YW, X0:X0+XW]
  else:
    X0 = 0.
    Y0 = 0.
    XW = out.shape[0]
    YW = out.shape[1] 
  #print out.shape

   
  numpy.savetxt(fileprefix+'.dat', out, fmt='%d')
  falsecolor.savepng( out, out.min(), out.max(), falsecolor.my_rainbow, fileprefix, dpi)
  
  return out

def get_counts( out , box):
  #print out.max()
  col = out.max(axis=0).argmax()
  row = out.max(axis=1).argmax()
  #print "Max at (%d,%d)" % ( row,col)
  #print out[row,col]

  #print args.SIZE
  br0 = 1+row+numpy.ceil(-box/2)
  br1 = 1+row+numpy.ceil(box/2)
  bc0 = 1+col+numpy.ceil(-box/2)
  bc1 = 1+col+numpy.ceil(box/2)
 
  bragg = out[ br0:br1 , bc0:bc1]
  patch = out[  0: br1-br0, 0: bc1-bc0]
  
  #print bragg
  #print patch 
 
  braggcounts = numpy.sum(numpy.sum(bragg))
  patchcounts = numpy.sum(numpy.sum(patch))
  return braggcounts, patchcounts

def analyze_path( mantapath , pngprefix, roi, box):
  print "Analyzing %s" % e
  shotnum =  os.path.basename( mantapath ).split('atoms.manta')[0]
  shotnum = "%04d" % int(shotnum)
  atomsfile = shotnum + 'atoms.manta'
  pngprefix = pngprefix + shotnum + '_bragg_'
  out = makepng( atomsfile , "PHC", 75, pngprefix, roi)

  braggcounts, patchcounts = get_counts(out, box)  

  inifile = "report" + shotnum + ".INI"
  report = ConfigObj(inifile)
 
  report['BRAGG'] = {} 
  report['BRAGG']['bragg'] = braggcounts
  report['BRAGG']['patch'] = patchcounts
  
  report.write()
 
  print "%s -->  BraggCounts =  %6d ,  BraggPatch = %6d" % (shotnum, braggcounts, patchcounts) 
  


# --------------------- MAIN CODE  --------------------#


if __name__ == "__main__":
  parser = argparse.ArgumentParser('braggData.py')
  parser.add_argument('SIZE', action="store", type=int, help='size in pixels of bragg peak region')
  parser.add_argument('-r', action="store", dest='ROI', type=str, help='XO,Y0,XW,YW region of intereset')
  parser.add_argument('-d', action = "store", dest='DIR', type=str, help="path of directory for output png file")
  parser.add_argument('--backwards', action = "store_true", dest='BACKWARDS', help="analyze files already on disk starting with the last one.  prompt at each")
 
  args = parser.parse_args()
  #print type(args)
  #print args

  import monitordir
  import os

  if args.DIR:
     pngprefix = args.DIR  
  else:
     pngprefix = ''
 
  list = glob.glob( os.getcwd() +  '/????atoms.manta' )
  list.sort()
  e=list[-1]
  analyze_path( e, pngprefix, args.ROI, args.SIZE)

  if args.BACKWARDS:
    list = glob.glob( os.getcwd() +  '/????atoms.manta' )
    list.sort()
    list.reverse()
    print list
    for e in list:
      analyze_path( e, pngprefix, args.ROI, args.SIZE)
      cont = raw_input("Continue backwards? [y/n]")
      if cont != 'y':
        exit(1)
   
  while (1):
    new = monitordir.waitforit( os.getcwd(), '/????atoms.manta' )
    for e in new:
      analyze_path( e, pngprefix, args.ROI, args.SIZE)
