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

from braggEigen import eigenclean_Bragg

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

ANDOR1EIGEN_sec = 'ANDOR1EIGEN'
ANDOR2EIGEN_sec = 'ANDOR2EIGEN'
MANTAEIGEN_sec = 'MANTAEIGEN'
HHHEIGEN_sec = 'HHHEIGEN'

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
      atoms_M = numpy.zeros_like( atoms)
      noatoms_M = numpy.zeros_like( atoms)
      nshots = 0
      binning_M =  (1038/atoms.shape[0], 1390/atoms.shape[1]) 
      
      for shotnum in args.shots: 
        try:
          atoms     = numpy.loadtxt( shotnum + 'atoms.manta')
          noatoms   = numpy.loadtxt( shotnum + 'noatoms.manta')
          atoms_M = atoms_M + atoms
          noatoms_M = noatoms_M + noatoms
          nshots = nshots + 1
        except:
          print "Error obtaining data from shot #%d" % shotnum
      
      atoms_M = atoms_M / nshots
      noatoms_M = noatoms_M / nshots 
   
    else:
      atoms     = numpy.loadtxt( shotnum + 'atoms.manta')
      noatoms   = numpy.loadtxt( shotnum + 'noatoms.manta')
      atoms_M = atoms
      noatoms_M = noatoms
      binning_M =  (1038/atoms.shape[0], 1390/atoms.shape[1])
    MANTA = True
    #MANTA = False

  except:
    MANTA = False

  # ANDOR1 pictures
  if not args.ignoreA1:
      try : 
        if average:
          shotnum = args.shots[0]
          atoms     = pyfits.open( shotnum + 'atoms.fits')[0].data[0]
          atoms_A = numpy.zeros_like( atoms)
          noatoms_A = numpy.zeros_like( atoms)
          nshots = 0
          binning_A =  (512/atoms.shape[0], 512/atoms.shape[1]) 
          
          for shotnum in args.shots: 
            try:
              atoms     = pyfits.open( shotnum + 'atoms.fits')[0].data[0]
              noatoms   = pyfits.open( shotnum + 'noatoms.fits')[0].data[0]
              atoms_A = atoms_A + atoms
              noatoms_A = noatoms_A + noatoms
              nshots = nshots + 1
            except:
              print "Error obtaining data from shot #%d" % shotnum
        
          atoms_A = atoms_A / nshots
          noatoms_A = noatoms_A / nshots 
        else:
          atoms     = pyfits.open( shotnum + 'atoms.fits')[0].data[0]
          noatoms   = pyfits.open( shotnum + 'noatoms.fits')[0].data[0]
          atoms_A = atoms
          noatoms_A = noatoms
          binning_A =  (512/atoms.shape[0], 512/atoms.shape[1])
        ANDOR1 = True
      except Exception as e:
        print e
        ANDOR1 = False
  else:
      ANDOR1 = False



  # ANDOR2 pictures
  try : 
    if average:
      shotnum = args.shots[0]
      atoms     = pyfits.open( shotnum + 'atoms_andor2.fits')[0].data[0]
      atoms_2 = numpy.zeros_like( atoms)
      noatoms_2 = numpy.zeros_like( atoms)
      nshots = 0
      binning_2 =  (512/atoms.shape[0], 512/atoms.shape[1]) 
      
      for shotnum in args.shots: 
        try:
          atoms     = pyfits.open( shotnum + 'atoms_andor2.fits')[0].data[0]
          noatoms   = pyfits.open( shotnum + 'noatoms_andor2.fits')[0].data[0]
          atoms_2 = atoms_2 + atoms
          noatoms_2 = noatoms_2 + noatoms
          nshots = nshots + 1
        except:
          print "Error obtaining data from shot #%d" % shotnum
    
      atoms_2 = atoms_2 / nshots
      noatoms_2 = noatoms_2 / nshots 
    else:
      atoms     = pyfits.open( shotnum + 'atoms_andor2.fits')[0].data[0]
      noatoms   = pyfits.open( shotnum + 'noatoms_andor2.fits')[0].data[0]
      atoms_2 = atoms
      noatoms_2 = noatoms
      binning_2 =  (512/atoms.shape[0], 512/atoms.shape[1]) 
    print atoms.shape 
    ANDOR2 = True
  except Exception as e:
    print e
    ANDOR2 = False


  #######################################################
  #### RESULTS ARE OBTAINED FOR ALL THREE CAMERAS
  #######################################################
  if average:
    eigenprefix = rangestr
  else:
    eigenprefix = shotnum

  results = []
  if MANTA and (args.only == None or 'manta' in args.only):
    results_M = eigenclean_Bragg(atoms_M, noatoms_M, args.discard_M, args.roi_M, args.pixels_M, shotnum, eigenprefix+'_manta', 'manta',Nbgs=args.nbgs)
    results_M['section'] = MANTAEIGEN_sec
    results.append(results_M)
    print 'MANTA\t',
    print bcolors.WARNING + "%7.1f +/- %6.1f "%(results[-1]['signal'],results[-1]['error']) + bcolors.ENDC,
    print "mean_out=%.2f" % results[-1]['mean_out'],
    print "stdev_out=%.2f" % results[-1]['stdev_out']
  elif not MANTA :
    print 'MANTA\t  no pictures' 
  else: 
    print 'MANTA\t  skipping'


  if ANDOR1 and (args.only == None or 'andor' in args.only):
    results_A= eigenclean_Bragg(atoms_A, noatoms_A, args.discard_A, args.roi_A, args.pixels_A, shotnum, eigenprefix+'_andor', 'andor',Nbgs=args.nbgs) 
    results_A['section'] = ANDOR1EIGEN_sec 
    results.append(results_A)
    print 'ANDOR1\t',
    print bcolors.WARNING + "%7.1f +/- %6.1f "%(results[-1]['signal'],results[-1]['error']) + bcolors.ENDC,
    print "mean_out=%.2f" % results[-1]['mean_out'],
    print "stdev_out=%.2f" % results[-1]['stdev_out']
  elif not ANDOR1:
    print 'ANDOR1\t  no pictures'
  else:
    print 'ANDOR1\t  skipping'
  

  if ANDOR2 and (args.only == None or 'andor2' in args.only):
    results_2 = eigenclean_Bragg(atoms_2, noatoms_2, args.discard_2, args.roi_2, args.pixels_2, shotnum, eigenprefix+'_andor2', 'andor2',Nbgs=args.nbgs) 
    results_2['section'] = ANDOR2EIGEN_sec 
    results.append(results_2)
    print 'ANDOR2\t',
    print bcolors.WARNING + "%7.1f +/- %6.1f "%(results[-1]['signal'],results[-1]['error']) + bcolors.ENDC,
    print "mean_out=%.2f" % results[-1]['mean_out'],
    print "stdev_out=%.2f" % results[-1]['stdev_out']
  elif not ANDOR2:
    print 'ANDOR2\t  no pictures'
  else:
    print 'ANDOR2\t  skipping'


  if results == []:
     return

  if average:
      inifile = "report" + rangestr + ".INI"
      print "Average results will be stored in:\n\t" + inifile
      if not os.path.exists(inifile):
          open(inifile,'w').close()  
  else:
      inifile = "report" + shotnum + ".INI"

  report = ConfigObj(inifile)
  for r in results:
    report[r['section']] = {}
    for key in r.keys():
        if key in ['signal','error','stdev_out','mean_out', 'signalD', 'signalS']:
          report[r['section']][key] = r[key]
  report.write()

 # Here I was saving the averaged pictures to file
 # if average:
 #     print "Saving shots to %s" % rangestr
 #     if MANTA:
 #       numpy.savetxt( rangestr + '_MANTA.dat', results_M[0]) 
 #     if ANDOR1:
 #       numpy.savetxt( rangestr + '_ANDOR1.dat', results_A[0]) 
 #     if ANDOR2:
 #       numpy.savetxt( rangestr + '_ANDOR2.dat', results_2[0]) 

  #print results 
  if ANDOR1 and ANDOR2:
    for r in results:
      if r['section'] == ANDOR1EIGEN_sec:
        andor1signal =  ufloat( ( r['signal'], r['error'] ) )
      if r['section'] == ANDOR2EIGEN_sec:
        andor2signal =  ufloat( ( r['signal'], r['error'] ) )
    #print andor1signal
    #print andor2signal
    andor2norm =  andor2signal / andor1signal
    print "HHH ",
    print bcolors.OKGREEN + "%3.3f +/- %3.3f "% ( andor2norm.nominal_value, andor2norm.std_dev() )  + bcolors.ENDC
    if average:
      inifile = "report" + rangestr + ".INI"
    else:
      inifile = "report" + shotnum + ".INI"
    report = ConfigObj(inifile)
    report[HHHEIGEN_sec] = {}
    report[HHHEIGEN_sec]['andor2norm'] = andor2norm.nominal_value
    report[HHHEIGEN_sec]['andor2norm_err'] = andor2norm.std_dev() 
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

  #if average:
  #  statimage.maskedimages_all( results, args.path + rangestr + '_detail.png' )
  #else: 
  #  statimage.maskedimages_all( results, args.path + shotnum + '_detail.png' ) 
 
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

  
  # Parameters related to selecting shots for analysis
  # If non is specified the default is to monitor the folder 
  # for data.  

  parser.add_argument('--range', action = "store", \
         dest='range', help="analyze files in the specified range.  \
                             DO NOT prompt at each")

  parser.add_argument('--average', action = "store_true", \
          help="if used with range, will run the analysis on an average of all shots in range." )

  parser.add_argument('--only', action = "store", nargs = '*',\
          help="allows selecting cameras to peform the analysis. useful when reanalyzing data only for one camera." )
 
  parser.add_argument('--ignoreA1', action = "store_true", \
          help="Andor1 pictures will not be analyzed" )

  parser.add_argument('--nbgs', action="store", type = int, default=30,\
          help="Number of backgrouds to be used by eigen")

  args = parser.parse_args()

  print args.range


  # The default values for the parameters are set here

  # MANTA
  dx = -75
  dy = -5
  args.roi_M = [715+dx,452+dy,65,65]
  args.discard_M = [600+dx,350+dy,300,300]
  args.pixels_M=[]
  for i in range(733+dx-13,763+dx+13):
    for j in range(467+dy-13,497+dy+13):
      args.pixels_M.append( (i,j) )

  # ANDOR1
  #args.roi_A = [228,225,48,48]
  dxA1 = -19 #-9
  dyA1 = 8
  args.roi_A = [219+dxA1,217+dyA1,48+42,48+42]
  args.discard_A = [100+dxA1,100+dyA1,312,312]
  args.pixels_A=[]

  # bigger
  for i in range(228+dxA1,300+dxA1):
    for j in range(226+dyA1,296+dyA1):
      args.pixels_A.append( (i,j) )

  
  # ANDOR2
  # 8x8 BINNING Data on 130406
  #args.roi_2 = [30,26,8,8] 
  #args.roi_2 = [28,24,12,12] #UP TO July 30
  #args.roi_2 = [28,24,12,12] 
  dyA2=17	
  dxA2=-2	
  args.roi_2 = [28+dxA2,24+dyA2,12,12] 
  args.discard_2 = [8,8,48,48]
  #
  # 2x2
  #args.pixels_2=[ (33,30), (33,31), (32,30), (32,31) ]  # Data on 130612
  #
  #args.pixels_2=[ (33,30), (33,29), (33,31), (32,30), (34,30) ] # Used on 130613 when trying NO IR filter
  #args.pixels_2=[ (33,30), (33,31), (32,30), (32,31)]  # Data on 130612
  #
  # 2x3
  ##args.pixels_2=[ (33,30),(33,29), (33,31), (32,29), (32,30), (32,31) ]
  #
  # 4x5
  #args.pixels_2=[]
  #for i in range(31,35):
  #  for j in range(28,33):
  #     args.pixels_2.append( (i,j) )
  # 
  # 8x8
  #args.pixels_2=[]
  #for i in range(29,37):
  #  for j in range(27,35):
  #     args.pixels_2.append( (i,j) ) 
  # 8x8
  args.pixels_2=[]
  for i in range(32+dxA2,37+dxA2):
    for j in range(27+dyA2,32+dyA2):
       args.pixels_2.append( (i,j) ) 
  # 
  # 4x4
  #args.pixels_2=[]
  #for i in range(31,35):
  #  for j in range(29,33):
  #     args.pixels_2.append( (i,j) ) 
 
 
  # 1x1 BINNING Data on 130521
  #args.roi_2 = [256,236,20,20]
  #args.pixels_2=[]
  #for i in range(261,270):
  #  for j in range(242,251):
  #    args.pixels_2.append( (i,j) ) 
   
  #args.roi_2 = [16,15,4,4] # 16x16 BINNING
  #args.roi_2 = [59,54,16,16] # 4x4 BINNING

  import monitordir
  import os
  import time
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
    

  path = os.getcwd()
  exprs = ['/????atoms.manta', '/????atoms.fits', '/????atoms_andor2.fits'] 
  
  last =  monitordir.get_last_shot( path, exprs  )
  #print last
  current = monitordir.get_all_shots( path, exprs) 
  #print current
  analyze_shot( last, args)

  from multiprocessing import Pool

  def a( x ):
      return analyze_shot( x, args) 

  def parallel_analyze(new):
      pool = Pool(4)
      ret = pool.map( a, new ) 
      pool.close()
      pool.join()
      return ret

  while (1):
    new = monitordir.waitforit_list( path, exprs ,\
          current = current )
    print
    print "List of shots to analyze:", new
    current = monitordir.get_all_shots( path, exprs) 
    time.sleep(1)
    #parallel_analyze( new ) 
    for e in new:
      analyze_shot( e, args)



