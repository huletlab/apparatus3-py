#!/usr/bin/python

import argparse
import sys 
import pyfits
import wx
import numpy
import glob
import hashlib

from configobj import ConfigObj

sys.path.append('/lab/software/apparatus3/bin/py')
import falsecolor
import gaussfitter
import qrange
import statimage

from uncertainties import ufloat

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

    def disable(self):
        self.HEADER = ''
        self.OKBLUE = ''
        self.OKGREEN = ''
        self.WARNING = ''
        self.FAIL = ''
        self.ENDC = ''


#######################################################
#
#  get_counts  
#
#######################################################
def get_counts( out , info, boxsize):
  #print out.max()
  col = out.max(axis=0).argmax()
  row = out.max(axis=1).argmax()

  
  if boxsize == info["signalsize"]:
      verbose = True
  else:
      verbose = False

  if numpy.linalg.norm( [info["CX"]-info["X0"]-row, info["CY"]-info["Y0"]-col] ) > 2.:
      # When max is not at the user specified center
      # it sets row,col at the user specified center
      row = info["CX"]-info["X0"]
      col = info["CY"]-info["Y0"]

  #if verbose:
  #    print " c=(%d,%d)" %  ( row+info["X0"],col+info["Y0"]),


  # Define region that will be masked out to determine the offset
  #br0 = row+numpy.ceil(-boxsize/2)
  #br1 = row+numpy.ceil(boxsize/2)
  #bc0 = 1+col+numpy.ceil(-boxsize/2)
  #bc1 = 1+col+numpy.ceil(boxsize/2)
  br0 = row - boxsize/2 
  br1 = row + boxsize/2 + boxsize%2
  bc0 = col - boxsize/2
  bc1 = col + boxsize/2 + boxsize%2
 
  signal = out[ br0:br1 , bc0:bc1]
  #if verbose:
  #  print "\nsignal.shape =\n", signal.shape


  mask = numpy.zeros_like(out)
  mask[ br0:br1, bc0:bc1 ] = 1
  masked = numpy.ma.MaskedArray(out, mask= mask)
  maskedpixels = masked.compressed()
  offset = numpy.mean(maskedpixels)
    

  signalcounts = numpy.sum(numpy.sum(signal))
  npix = signal.size
  bgndcounts = npix*offset

  # The signal is obtained as 
  #
  # signal = \sum_{i} (  photons_{i} + atoms_{i} - noatoms_{i} ) 
  #
  # This is just the sum of a bunch of independent random variables, so the variance is the 
  # sum of variances:
  #
  # var(signal) = \sum_{i} ( var( photons_{i} ) + var ( atoms_{i} - noatoms_{i} ) ) 
  #
  # var( atoms_{i} - noatoms_{i} ) is just the square of the standard deviation
  # of all the pixels in the maskedpixels, so
  #
  # var(signal) = \sum_{i} ( var( photons_{i} ) + stdev(maskedpixels)^2 ) 
  #
  # We neglect the photon shot noise:
  #
  # var(signal) = \sum_{i} stdev(maskedpixels)^2  = npixels * stdev(maskedpixels)^2 
  # 
  # The error in the signal is  sqrt( var(signal) ) = sqrt(npixels) * stdev(maskedpixels) 

  
  signalerror = numpy.sqrt(npix)*numpy.std(maskedpixels) 

  results={}
  results["signalregion"]=[br0,br1,bc0,bc1]  
  results['col'] = col
  results['row'] = row

  results['peak_raw'] = out[row,col]
  results['peak_offset'] = offset
  results['peak_net'] = out[row,col] - offset

  results['sum_raw_%d'%boxsize] = signalcounts
  results['sum_offset_%d'%boxsize] = bgndcounts
  results['sum_net_%d'%boxsize] = signalcounts - bgndcounts
  results['sum_net_%d_err'%boxsize] = signalerror
  
  if verbose:
      print bcolors.WARNING + "%7.1f +/- %6.1f "%(results['sum_net_%d'%boxsize],results['sum_net_%d_err'%boxsize]) + bcolors.ENDC,
      print "usr@(%d,%d)=%d" % ( info["CX"], info["CY"] , info["peak_cts"]),
      print "  max@(%d,%d)=%d" % ( row+info["X0"],col+info["Y0"], out[row,col]),

  if verbose:
    print "  peak:%.1f[%d,%.1f]  sum(%d):%.1f[%d,%.1f]" % \
    (results['peak_net'], results['peak_raw'], results['peak_offset'],\
     signal.shape[0],\
     results['sum_net_%d'%boxsize], results['sum_raw_%d'%boxsize], results['sum_offset_%d'%boxsize]),
    print "  offset=%.3f" % offset
          
  return results, signal, masked 


#######################################################
#
#  cop_and_sum
#
#######################################################
def crop_and_sum( image, args, shotnum, section ):

  if section == 'MANTABRAGG':
    c = args.c_M
    cropsize = args.size_M
    signalsize = args.signalsize_M 
  elif section == 'ANDOR1':
    c = args.c_A
    cropsize = args.size_A
    signalsize = args.signalsize_A
  elif section == 'ANDOR2':
    c = args.c_2
    cropsize = args.size_2
    signalsize = args.signalsize_2

 
  info = {}
  try:
    CX = float(c.split(',')[0])   
    CY = float(c.split(',')[1])   
    info["peak_cts"] = image[CY,CX]

    # Here the full frame is cropped
    hs = cropsize / 2
    off = 0 
    out = image[ CY-hs-off:CY+hs-off, CX-hs-off:CX+hs-off ]

    info["camera"] = section
    info["X0"] = CX-hs-off
    info["Y0"] = CY-hs-off
    info["CX"] = CX
    info["CY"] = CY
    info["c"] = c
    info["cropsize"] = cropsize
    info["signalsize"] = signalsize

  except:
    print "\n...  (%s) Could not crop to specified center and size" % section
    print "...  c = (%s),  size = %d" % (c, cropsize)
    exit(1)

  print "%s\t" % section,
  inifile = "report" + shotnum + ".INI"
  report = ConfigObj(inifile)
  report[section] = {}
  for boxsize in ( 2., 3., 5., 8., 12., 16., 20., 24.):
    # r is a dictionary with the results of get_counts
    r, s, m = get_counts(out, info, boxsize) 

    if boxsize == info["signalsize"]:
        info["signalregion"] = r["signalregion"]

        info["signal"] = r['sum_net_%d'%boxsize]
        report[section]["signal"] = r['sum_net_%d'%boxsize]

        info["error"] = r['sum_net_%d_err'%boxsize] 
        report[section]["error"] = r['sum_net_%d_err'%boxsize] 
  
        info["signalpx"] = s
        info["maskedpx"] = m

    for key in r.keys():
        report[section][key] = r[key]

    report.write()


  return out, info


#######################################################
#
#  analyze_shot
#
#######################################################
def analyze_shot( shot , args, average=False):
  shotnum = "%04d" % shot
  print bcolors.HEADER + "\n%s" % shotnum + bcolors.ENDC

  if average:
    print "Averaging shots in ", args.range
    print args.shots
    rangestr = args.range 
    rangestr = rangestr.replace('-','m')
    rangestr = rangestr.replace(':','-')
    rangestr = rangestr.replace(',','_')
    if len(rangestr) > 16:
        rangestr = hashlib.sha224(rangestr).hexdigest()[:16]
    
    rangestr = "avg_" + rangestr



  #######################################################
  #### PICTURES ARE COLLECTED FROM ALL THREE CAMERAS
  #######################################################

  # MANTA pictures
  try :

    if average:
      shotnum = args.shots[0]
      atoms     = numpy.loadtxt( shotnum + 'atoms.manta')
      out_M = numpy.zeros_like( atoms)  
      nshots = 0
      binning_M =  (1038/atoms.shape[0], 1390/atoms.shape[1]) 
      
      for shotnum in args.shots: 
        try:
          atoms     = numpy.loadtxt( shotnum + 'atoms.manta')
          noatoms   = numpy.loadtxt( shotnum + 'noatoms.manta')
          out_M = out_M + atoms - noatoms 
          nshots = nshots + 1
        except:
          print "Error obtaining data from shot #%d" % shotnum
    
      out_M = out_M / nshots
   
    else:

      atoms     = numpy.loadtxt( shotnum + 'atoms.manta')
      noatoms   = numpy.loadtxt( shotnum + 'noatoms.manta')
      out_M = atoms - noatoms 
      binning_M =  (1038/atoms.shape[0], 1390/atoms.shape[1])
    atoms_M = atoms
    noatoms_M = noatoms 
    MANTA = True

  except:
    MANTA = False

  # ANDOR1 pictures
  try : 
    if average:
      shotnum = args.shots[0]
      atoms     = pyfits.open( shotnum + 'atoms.fits')[0].data[0]
      out_A = numpy.zeros_like( atoms)  
      nshots = 0
      binning_A =  (512/atoms.shape[0], 512/atoms.shape[1]) 
      
      for shotnum in args.shots: 
        try:
          atoms     = pyfits.open( shotnum + 'atoms.fits')[0].data[0]
          noatoms   = pyfits.open( shotnum + 'noatoms.fits')[0].data[0]
          out_A = out_A + atoms - noatoms 
          nshots = nshots + 1
        except:
          print "Error obtaining data from shot #%d" % shotnum
    
      out_A = out_A / nshots
    else:
      atoms     = pyfits.open( shotnum + 'atoms.fits')[0].data[0]
      noatoms   = pyfits.open( shotnum + 'noatoms.fits')[0].data[0]
      out_A = atoms - noatoms
      binning_A =  (512/atoms.shape[0], 512/atoms.shape[1])
    atoms_A = atoms
    noatoms_A = noatoms 
    ANDOR1 = True
  except:
    ANDOR1 = False

  # ANDOR2 pictures
  try : 
    if average:
      shotnum = args.shots[0]
      atoms     = pyfits.open( shotnum + 'atoms_andor2.fits')[0].data[0]
      out_2 = numpy.zeros_like( atoms)  
      nshots = 0
      binning_2 =  (512/atoms.shape[0], 512/atoms.shape[1]) 
      
      for shotnum in args.shots: 
        try:
          atoms     = pyfits.open( shotnum + 'atoms_andor2.fits')[0].data[0]
          noatoms   = pyfits.open( shotnum + 'noatoms_andor2.fits')[0].data[0]
          out_2 = out_2 + atoms - noatoms 
          nshots = nshots + 1
        except:
          print "Error obtaining data from shot #%d" % shotnum
    
      out_2 = out_2 / nshots
    else:
      atoms     = pyfits.open( shotnum + 'atoms_andor2.fits')[0].data[0]
      noatoms   = pyfits.open( shotnum + 'noatoms_andor2.fits')[0].data[0]
      out_2 = atoms - noatoms 
      binning_2 =  (512/atoms.shape[0], 512/atoms.shape[1]) 
    atoms_2 = atoms
    noatoms_2 = noatoms 
    print atoms.shape 
    ANDOR2 = True
  except:
    ANDOR2 = False


  #######################################################
  #### RESULTS ARE OBTAINED FOR ALL THREE CAMERAS
  #######################################################
  results = []
  if MANTA:
    results_M  = crop_and_sum( out_M, args, shotnum, 'MANTABRAGG')
    results_M = list(results_M)
    results_M.append(atoms_M)
    results_M.append( noatoms_M)
    results.append(results_M)
    # All cropped data is saved in the data dir for averaging use 
    # 2013/04/09 not doing this anymore
    # numpy.savetxt(shot+'_bragg.dat', out, fmt='%d')
  else:
    print 'MANTA\t  no pictures'

  if ANDOR1:
    results_A = crop_and_sum( out_A, args, shotnum, 'ANDOR1')
    results_A = list(results_A)
    results_A.append(atoms_A)
    results_A.append(noatoms_A)
    results.append(results_A)
  else:
    print 'ANDOR1\t  no pictures'

  if ANDOR2:
    results_2 = crop_and_sum( out_2, args, shotnum, 'ANDOR2')
    results_2 = list(results_2)
    results_2.append(atoms_2)
    results_2.append(noatoms_2)
    results.append(results_2)
  else:
    print 'ANDOR2\t  no pictures'

  if results == []:
     return

  if average:
      print "Saving shots to %s" % rangestr
      if MANTA:
        numpy.savetxt( rangestr + '_MANTA.dat', results_M[0]) 
      if ANDOR1:
        numpy.savetxt( rangestr + '_ANDOR1.dat', results_A[0]) 
      if ANDOR2:
        numpy.savetxt( rangestr + '_ANDOR2.dat', results_2[0]) 

  if ANDOR1 and ANDOR2:
    for r in results:
      if r[1]['camera'] == 'ANDOR1':
        andor1signal =  ufloat( ( r[1]['signal'], r[1]['error'] ) )
      if r[1]['camera'] == 'ANDOR2':
        andor2signal =  ufloat( ( r[1]['signal'], r[1]['error'] ) )
    #print andor1signal
    #print andor2signal
    andor2norm =  andor2signal / andor1signal
    print "HHH ",
    print bcolors.OKGREEN + "%3.3f +/- %3.3f "% ( andor2norm.nominal_value, andor2norm.std_dev() )  + bcolors.ENDC
    inifile = "report" + shotnum + ".INI"
    report = ConfigObj(inifile)
    report['HHH'] = {}
    report['HHH']['andor2norm'] = andor2norm.nominal_value
    report['HHH']['andor2norm_err'] = andor2norm.std_dev() 
    report.write()
     
  # MAKE PLOTS TO ILLUSTRATE ANALYSIS AND RESULTS
  # 
  # results is a list that contains results for each camera. 
  #
  # for each camera, the results object is another list, 
  # which contains: 
  # 
  #   results[0] = matrix with pixels in the camera ROI
  #   results[1] = dictionary with analysis information: 
  #                camera   = camera name 
  #                signalpx = matrix with pixels used to obtain signal sum
  #                maskedpx = matrix with pixels used to obtain offset
  # 
  
  if not os.path.exists(args.path):
    os.makedirs(args.path)

  if average:
    statimage.maskedimages_all( results, args.path + rangestr + '_detail.png' )
  else: 
    statimage.maskedimages_all( results, args.path + shotnum + '_detail.png' ) 
 
  #pngprefix = args.path + shotnum + '_bragg' 
  #falsecolor.inspecpng( [out], \
  #                      r['row'], r['col'], out.min(), out.max(), \
  #                      falsecolor.my_grayscale, \
  #                      pngprefix, 100, origin = 'upper' , \
  #                      step=True, scale=10, \
  #                      interpolation='nearest', \
  #                      extratext='')

  
  return


# --------------------- MAIN CODE  --------------------#


if __name__ == "__main__":
  parser = argparse.ArgumentParser('braggData.py')

  # M is for manta
  # A is for Andor1
  # 2 is for Andor2

  
  # MANTA
  parser.add_argument('--size_M', \
         action="store", type=int, \
         help='size in px of crop region.')

  parser.add_argument('--signalsize_M', \
         action="store", type=int, \
         help='size in pixels of bragg peak.')

  parser.add_argument('-c_M', \
         action="store", type=str, \
         help='X,Y, of Bragg pixel.')


  # ANDOR
  parser.add_argument('--size_A', \
         action="store", type=int, \
         help='size in px of crop region.')

  parser.add_argument('--signalsize_A', \
         action="store", type=int, \
         help='size in pixels of bragg peak.')

  parser.add_argument('-c_A', \
         action="store", type=str, \
         help='X,Y, of Bragg pixel.')


  # ANDOR2
  parser.add_argument('--size_2', \
         action="store", type=int, \
         help='size in px of crop region.')

  parser.add_argument('--signalsize_2', \
         action="store", type=int, \
         help='size in pixels of bragg peak.')

  parser.add_argument('-c_2', \
         action="store", type=str, \
         help='X,Y, of Bragg pixel.')


  # Parameters related to selecting shots for analysis
  # If non is specified the default is to monitor the folder 
  # for data.  

  parser.add_argument('--range', action = "store", \
         dest='range', help="analyze files in the specified range.  \
                             DO NOT prompt at each")

  parser.add_argument('--average', action = "store_true", \
          help="if used with range, will run the analysis on an average of all shots in range." )
 
  args = parser.parse_args()

  print args.range


  # The default values for the parameters are set here

  # MANTA
  if not args.signalsize_M:
      args.signalsize_M = 12 
  if not args.size_M:
      args.size_M = 48
  if not args.c_M:
      args.c_M = '740,474'  # Last pixel where we saw counts

  # ANDOR
  if not args.signalsize_A:
      args.signalsize_A = 24
  if not args.size_A:
      args.size_A = 96
  if not args.c_A:
      args.c_A = '270,249'  # Last pixel where we saw counts
  
  # ANDOR2

  # WITHOUT BINNING
  #if not args.signalsize_2:
  #    args.signalsize_2 = 20
  #if not args.size_2:
  #    args.size_2 = 96
  #if not args.c_2:
  #    args.c_2 = '266,246'  # Last pixel where we saw counts

  # WITH BINNING 8x8
  #if not args.signalsize_2:
  #    args.signalsize_2 = 3
  #if not args.size_2:
  #    args.size_2 = 12
  #if not args.c_2:
  #    args.c_2 = '34,31'  # Last pixel where we saw counts

  # WITH BINNING 16x16
  if not args.signalsize_2:
      args.signalsize_2 = 2
  if not args.size_2:
      args.size_2 = 4
  if not args.c_2:
      args.c_2 = '17,16'  # Last pixel where we saw counts

  # WITH BINNING 4x4
  #if not args.signalsize_2:
  #    args.signalsize_2 = 5
  #if not args.size_2:
  #    args.size_2 = 24
  #if not args.c_2:
  #    args.c_2 = '67,62'  # Last pixel where we saw counts

  import monitordir
  import os

  #Set the directory where png's will be saved
  args.path = 'braggdata/' 
 
  
  if args.range != None:
    print "average = ", args.average
    shots = qrange.parse_range( args.range)
    args.shots = shots
    if args.average == False:
      for shot in shots:
         analyze_shot( int(shot), args)
      print ""
      exit(0)
    else:
      analyze_shot( int(shots[0]), args, average=True)
      print ""
      exit(0)
    

  last =  monitordir.get_last_shot( os.getcwd(),  ['/????atoms.manta', '/????atoms.fits', '????atoms_andor2.fits'] )
  analyze_shot( last, args)

  while (1):
    new = monitordir.waitforit_list( os.getcwd(), ['/????atoms.manta', '/????atoms.fits', '????atoms_andor2.fits'] )
    for e in new:
      analyze_shot( e, args)



