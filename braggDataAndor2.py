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
import ast
import pprint
import statimage



def importfits(shot):
  atoms     = pyfits.open( shot + 'atoms_andor2.fits')[0].data[0]
  noatoms   = pyfits.open( shot + 'noatoms_andor2.fits')[0].data[0]
  return atoms, noatoms


def get_counts( out , args, boxsize):
  box = boxsize
  #print out.max()
  col = out.max(axis=0).argmax()
  row = out.max(axis=1).argmax()


  if box == args.braggpixels:
      print "usr@(%d,%d)=%d" % ( args.CX, args.CY , args.Ccts),
      print "  max@(%d,%d)=%d" % ( row+args.X0,col+args.Y0, out[row,col]),

  if numpy.linalg.norm( [args.CX-args.X0-row, args.CY-args.Y0-col] ) > 2.:
      # When max is not at the user specified center
      # it sets row,col at the user specified center
      row = args.CX-args.X0
      col = args.CY-args.Y0

  if box == args.braggpixels:
      print " c=(%d,%d)" %  ( row+args.X0,col+args.Y0),


  #print args.SIZE
  br0 = 1+row+numpy.ceil(-box/2)
  br1 = 1+row+numpy.ceil(box/2)
  bc0 = 1+col+numpy.ceil(-box/2)
  bc1 = 1+col+numpy.ceil(box/2)
 
  bragg = out[ br0:br1 , bc0:bc1]
  #print "bragg.shape =", bragg.shape

  mask = numpy.ones( out.shape)
  mask[ br0:br1, bc0:bc1 ] = 0.

  offset = numpy.sum( mask * out ) \
           / ( out.shape[0]*out.shape[1] - bragg.shape[0]*bragg.shape[1] ) 
 
  patch = numpy.ones( out.shape) * offset 

  braggcounts = numpy.sum(numpy.sum(bragg))
  patchcounts = bragg.shape[0]*bragg.shape[1]*offset

  results={}
  results['col'] = col
  results['row'] = row

  results['peak_raw'] = out[row,col]
  results['peak_offset'] = offset
  results['peak_net'] = out[row,col] - offset

  results['sum_raw_%d'%box] = braggcounts
  results['sum_offset_%d'%box] = patchcounts
  results['sum_net_%d'%box] = braggcounts - patchcounts
  
  if box == args.braggpixels:
    print "  peak:%.1f[%d,%.1f]  sum(%d):%.1f[%d,%.1f]" % \
    (results['peak_net'], results['peak_raw'], results['peak_offset'],\
     bragg.shape[0],\
     results['sum_net_%d'%box], results['sum_raw_%d'%box], results['sum_offset_%d'%box]),
    print "  offset=%.3f" % offset
          

  return results


def analyze_stats( Z, stats):
  try:
    stats = ast.literal_eval(stats)
  except:
    print "Error parsing stats ROIs (regions of interest):"
    pprint.pprint( stats) 
    return
  statimage.statimage( Z, stats) 

 



def analyze_path( shot , args):

  atoms, noatoms = importfits(shot)
  out = atoms-noatoms
  
  shotnum = '%04d' % int(shot)
 
  if args.stats:
    analyze_stats( out, args.stats) 

  try:
    CX = float(args.c.split(',')[0])   
    CY = float(args.c.split(',')[1])   
    args.Ccts = out[CY,CX]
    hs = args.size / 2
    off = 0 
    out = out[ CY-hs-off:CY+hs-off, CX-hs-off:CX+hs-off ]
    args.X0 = CX-hs-off
    args.Y0 = CY-hs-off
    args.CX = CX
    args.CY = CY
  except:
    print "...Could not crop to specified center and size"
    print "...  c = (%s),  size = %d" % (args.c, args.size)
    exit(1)

  #All cropped data is saved in the data dir for averaging use
  numpy.savetxt(shot+'_bragg.dat', out, fmt='%d')

  #Results
  center = (CX,CY)
  inifile = "report" + shotnum + ".INI"
  report = ConfigObj(inifile)
  report['BRAGG'] = {}
  for boxsize in (2. ,4., 6., 8.):
    r = get_counts(out, args, boxsize) 
    for key in r.keys():
        report['BRAGG'][key] = r[key]
    report.write()

  if not os.path.exists(args.path):
    os.makedirs(args.path)
 
  pngprefix = args.path + shotnum + '_bragg' 
  falsecolor.inspecpng( [out], \
                        r['row'], r['col'], out.min(), out.max(), \
                        falsecolor.my_grayscale, \
                        pngprefix, 100, origin = 'lower' , \
                        step=True, scale=10, \
                        interpolation='nearest', \
                        extratext='')

  
  return


# --------------------- MAIN CODE  --------------------#


if __name__ == "__main__":
  parser = argparse.ArgumentParser('braggDataAndor2.py')

  parser.add_argument('--size', \
         action="store", type=int, \
         help='size in px of crop region.  default = 24')

  parser.add_argument('--braggpixels', \
         action="store", type=int, \
         help='size in pixels of bragg peak. default = 4')

  parser.add_argument('-c', \
         action="store", type=str, \
         help='X,Y, of Bragg pixel. default = 731,474')

  parser.add_argument('--backwards', \
         action = "store_true", dest='BACKWARDS', \
         help="analyze files already on disk starting with \
               the last one.  prompt at each")

  parser.add_argument('--forcebackwards', \
         action = "store_true", dest='FBACKWARDS', \
         help="analyze files already on disk starting with \
               the last one.  DO NOT prompt at each")

  parser.add_argument('--range', action = "store", \
         dest='range', help="analyze files in the specified range.  \
                             DO NOT prompt at each")

  parser.add_argument('--stats', action = "store", \
         dest='stats', help="Do image statistics on a set of ROIs in the image.\n"+\
                            "Syntax is --stats [[X0,Y0,XS,YS],[X0,Y0,XS,YS],...]")
 
  args,unknown = parser.parse_known_args()
 
  ## HANDLE CASE WHERE SHOT NUMBER IS INDICATED EXPLICITLY 
  if unknown != []:
     shot = int(unknown[0])
     if shot < 0 or shot >9999:
       print "Unrecognized arguments: ", unknown
     else:
       args.range = '%04d:%04d' % (shot,shot)
 
  ## DEFAULT PARAMETERS ARE SET HERE 
  if not args.braggpixels:
      args.braggpixels = 6  # Seems to work best
 
  if not args.size:
      args.size = 48  # 
                       # 
  if not args.c:
      args.c = '274,276'  # Last pixel where we saw counts
  

  import monitordir
  import os

  #Set the directory where png's will be saved
  args.path = 'braggdata/' 
 
  
  if args.range != None:
    shots = qrange.parse_range( args.range)
    list = glob.glob( os.getcwd() +  '/????atoms_andor2.fits' ) 
    list.sort()
    for e in list:
      eshot =  os.path.split(e)[1].split('atoms')[0]
      if eshot in shots:
          analyze_path( eshot, args)
    print ""
    exit(0)
    

  if args.BACKWARDS or args.FBACKWARDS:
    list = glob.glob( os.getcwd() +  '/????atoms.manta' )
    list.sort()
    list.reverse()
    #print list
    for e in list:
      try:
        analyze_path( e, args)
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
  print e
  analyze_path( e, args)

  while (1):
    new = monitordir.waitforit( os.getcwd(), '/????atoms.manta' )
    for e in new:
      analyze_path( e, args)



