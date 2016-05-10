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
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

import os
import cPickle
import time
import scipy
from scipy import linalg

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

############## FUNCTIONS ###############

def getimage( path, camera):
  if camera == 'andor2' or camera == 'andor':
    return pyfits.open( path  )[0].data[0]
  else:
    return numpy.loadtxt( path )

def getfitskey(  camera ):
    if camera == 'andor': 
      fitskey = '.fits'
    if camera == 'andor2':
      fitskey = '_andor2.fits'
    if camera == 'manta':
      fitskey = '.manta'
    return fitskey

 

def GetBackgrounds(Nbgs,shape,thisshot,camera, constbg=False):
  """This function looks in the current directory and gets the last
     Nbgs available backgrounds.  It checks that all backgrounds have
     the same shape.  It returns a list with the backgrounds, a 
     list with the corresponding shot number, and the shape of 
     the backgrounds. 
     """
  backgrounds = []
  bgshots = []
 
  fitskey = getfitskey( camera) 

  if thisshot != None:
    pwd = os.getcwd()
    noatoms_img = getimage( pwd + '/' + thisshot + 'noatoms' + fitskey , camera) 
    backgrounds.append( noatoms_img  )
    bgshots.append( thisshot )
  
  if Nbgs==0:
    return backgrounds, bgshots, shape

  # This is how it used to get done, it just picked the last 40 of whatever
  # it found in the current directory:
  #atoms      = glob.glob( os.getcwd() + '/????atoms' + fitskey )[-Nbgs:]

  atoms      = glob.glob( os.getcwd() + '/????atoms' + fitskey )
  shots = [ os.path.basename( a ).split('atoms')[0]  for a in atoms ]  
  #print "This is shot #", thisshot
  # Here, need to select the shots that are closest to thisshot and 
  # that match some basic report keys  
  
  # For this purpose, first sort the list by proximity to thisshot 
  keyfun = lambda x : min( (int(x) - int(thisshot))**2  , ( int(x)-10000 -int(thisshot) )**2 ) 
  shots = sorted( shots, key = keyfun )

  # Then start looking for the desired keys in 
  keys = [ ('ANDOR','exp') ,\
           ('DIMPLELATTICE','imgdet'),\
           ('DIMPLELATTICE','angle'),\
           ('DIMPLELATTICE','tof'),\
           ('DIMPLELATTICE','light'),\
           ('DIMPLELATTICE','camera'),\
           ('DIMPLELATTICE','bragg_pow'),\
           ('DIMPLELATTICE','andor2noatoms'),\
           ('DIMPLELATTICE','signal'),\
           ('DIMPLELATTICE','force_lcr3')\
         ]
  vals = []
  inifile = "report" + thisshot + ".INI"
  report = ConfigObj(inifile)
  for sec,key in keys:
      try:
        vals.append( report[ sec ][ key ] ) 
      except:
        emsg = "Error finding sec:key = %s:%s in:" % (sec, key) 
        print emsg
        print inifile
        raise Exception(msg)
        #exit(1)
         
  
  eigenshots = []
  for s in shots:
      if len(eigenshots) > Nbgs:
        break 
      inifile = "report" + s + ".INI"
      report = ConfigObj(inifile)
      matches = True
      for i,k in enumerate(keys):
        try:
          val = report[k[0]][k[1]]
        except:
          print "Error finding sec:key = %s:%s in:" % (k[0], k[1])
          print inifile
          exit(1)
 
        if report[k[0]][k[1]] != vals[i]:
          matches = False
      if matches:
        eigenshots.append( s )
  #print eigenshots

  atoms =   [ os.getcwd() + '/'  + s + 'atoms' + fitskey for s in eigenshots  ]
      
  atoms.sort()
  for img in atoms: 
    try:
      pwd = os.getcwd()
      shot = os.path.basename(img).rsplit('atoms')[0]
      if shot == thisshot:
        continue
      atoms_img = getimage( img , camera) 
      noatoms_img = getimage( pwd + '/' + shot + 'noatoms' + fitskey, camera) 
      if shape:
        if atoms_img.shape != shape:
          #print "error: found image of different shape %s" % img
          continue
      else:
        shape = atoms_img.shape
      backgrounds.append( noatoms_img  )
      bgshots.append( shot )
    except:
      print "error opening image : %s" % img
      exit(1)

  if constbg == True:
      mean = numpy.mean( numpy.array( backgrounds ) ) 
      print "Using constant eigen bgnd, mean = %.3f"%mean
      backgrounds.append( mean * numpy.ones(shape) ) 
      bgshots.append( 'flat' )  
 
  return backgrounds, bgshots, shape


def Bjk( mask, backgrounds ):
  """ This function calculates the entries of the Bjk matrix 
      which is used in the calculation of the coefficients
      that produce the eigenclean background.  The mask must
      be provided and it should be =1. in any part of the frame
      where there are atoms and 0. othewise
      This function returns the Bjk matrix and also the 
      normalization factor used to keep the entries of the Bjk
      matrix in the range (0,1). 
      The same normalization factor must be used to calculate
      the entries of Yj,  see below. 
  """
  start = time.time()
  lb = len(backgrounds)
  bjk = numpy.zeros( (lb,lb) ) 
  Bjk_dict = {}
  for j in range( lb ):
    for k in range( lb ):
      bjk[j,k] = numpy.sum( (1.-mask) * backgrounds[j] * backgrounds[k] )
      Bjk_dict[j,k] = bjk[j,k]
  bjk_max = numpy.max( bjk)
  bjk = bjk / bjk_max
  end = time.time()
  #print "Bjk evaluation time = %.2f seconds" % (end - start)
  return scipy.mat( bjk ), bjk_max


def Yj( mask, backgrounds, image, normfactor):
  """ This function calculates the entries of the Yj vector
      which is used in the calculation of the coefficients
      that produce the eigenclean background.  The mask 
      must be provided. 
      The normalization factor used to normalize Bjk must
      also be provided.  
  """
  start = time.time()
  lb = len(backgrounds)
  yj = numpy.zeros( lb ) 
  for j in range( lb ):
    yj[j] = numpy.sum( (1.-mask) * backgrounds[j] * image ) / normfactor
  end = time.time()
  #print "Yj evaluation time = %.2f seconds" % (end - start)
  return scipy.mat( yj ).T



def NewBackground( coefs, backgrounds):
  """ After the coefficients have been calculated, the 
      new eigenclean background is computed here. 
  """
  lb = len(backgrounds)
  if lb != len(coefs):
    print "error:  number of coefs and background images is not equal"
    exit(1)
  newbg = numpy.zeros( backgrounds[0].shape ) 
  for k in range( lb ):
    newbg = newbg + coefs[k]*backgrounds[k]
  return newbg



#######################################################
#
#  eigenclean_Bragg
#
#######################################################
def eigenclean_Bragg( atoms, noatoms, discard, roi, pixels, shot, fileprefix, camera, Nbgs=30, eigenverbose=False, constbg=False):
   rows = 3
   cols = 3

   discardmask = numpy.zeros_like(atoms)
   discardmask[ discard[1]:discard[1]+discard[3], discard[0]:discard[0]+discard[2] ] = 1.

   eigenmask = numpy.ones_like(atoms)
   eigenmask = eigenmask - discardmask
   for px in pixels:
       eigenmask[ (px[1],px[0]) ] = 1.
   #eigenmask[ roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2] ] = 1.
   
   mask = numpy.zeros_like(atoms)
   #for px in pixels:
   #    mask[ (px[1],px[0]) ] = 1.
   mask[ roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2] ] = 1.

   fig = plt.figure(figsize=(12.,12.))

   ax = plt.subplot2grid( (rows, cols), (0,0), rowspan=1, colspan=1)
   ax.set_title( 'atoms')
   colormap = matplotlib.cm.spectral
   im = ax.imshow( atoms, \
                    interpolation='nearest', \
                    cmap=colormap, \
                    origin='lower',\
                    vmax= 1.01*atoms.max(), \
                    vmin= atoms.min())
   plt.colorbar(im, use_gridspec=True)

   ax = plt.subplot2grid( (rows, cols), (0,1), rowspan=1, colspan=1)
   ax.set_title( 'noatoms')
   colormap = matplotlib.cm.spectral
   im = ax.imshow( noatoms, \
                    interpolation='nearest', \
                    cmap=colormap, \
                    origin='lower',\
                    vmax= 1.01*noatoms.max(), \
                    vmin= noatoms.min())
   plt.colorbar(im, use_gridspec=True)

   ####### SIMPLE SUBTRACTION AND SCALING  ALGORITHM #######

   #Subtraction
   diff = atoms-noatoms
   #Scaling
   masked_atoms = numpy.ma.MaskedArray(atoms, mask= eigenmask)
   masked_noatoms = numpy.ma.MaskedArray(noatoms, mask= eigenmask)
   atoms_outside = numpy.mean( masked_atoms.compressed() ) 
   noatoms_outside = numpy.mean( masked_noatoms.compressed() )
   eta = atoms_outside / noatoms_outside
   diffscale = atoms-eta*noatoms

   ####### EIGENCLEAN ALGORITHM #######
   #Nbgs=30
   bgs,bgshots,shape = GetBackgrounds(Nbgs, atoms.shape, shot, camera, constbg=constbg)
   B, normfactor =  Bjk( eigenmask, bgs )
   if eigenverbose:
     print
     print "# of backgrounds requested = %d" % Nbgs
     print "# of backgrounds used = %d" % len(bgs)
     print "pixels per background = %s" % str(bgs[0].shape)
     print "shape of B matrix = %s" % str(B.shape)
   image = atoms 
   y = Yj( eigenmask, bgs, image, normfactor)  
   if eigenverbose:
     print "shape of y vector = %s" % str(y.shape)
   t0 = time.time()
   c, resid, rank, sigma = linalg.lstsq( B, y)
   if eigenverbose:
     print "Coeffs evaluation time = %.2f seconds" % (time.time()-t0)
     print "maximum residual = %.3e" % numpy.max(y-B*c)
     print "\neigenclean background:\n" 
   if eigenverbose: 
     for i in range( c.shape[0]  ):
       out =  "% 2.3f * '%s'" % ( c[i], bgshots[i] ) 
       if bgshots[i] == shot :
         out = out + ' <--'
       print out
   t0 = time.time()
   newbg = NewBackground( c, bgs)
   if eigenverbose: 
     print "New background evaluation time = %.2f seconds" % (time.time()-t0)
   eigen = atoms-newbg

   savepath = 'braggeigen/' 
   if eigenverbose and True:
     import ccdHistogram
     ccdHistogram.histoeigen( bgs, atoms, newbg, eigen, shot = shot, savepath = savepath )

   ax = plt.subplot2grid( (rows, cols), (0,2), rowspan=1, colspan=1)
   ax.set_title( 'eigenface bg')
   colormap = matplotlib.cm.spectral
   im = ax.imshow( newbg, \
                    interpolation='nearest', \
                    cmap=colormap, \
                    origin='lower',\
                    vmax=  1.01*newbg.max(), \
                    vmin=  newbg.min())
   plt.colorbar(im, use_gridspec=True)

   ####### PLOT RESULTING BACKGROUNDS #######

   minscale = min( diff.min(), diffscale.min(), eigen.min() )
   maxscale = max( diff.max(), diffscale.max(), eigen.max() )*1.01
   
   #Subtraction
   ax = plt.subplot2grid( (rows, cols), (1,0), rowspan=1, colspan=1)
   ax.set_title( 'atoms-noatoms')
   colormap = matplotlib.cm.spectral
   m_diff = numpy.ma.MaskedArray( diff, mask=eigenmask)
   im = ax.imshow( m_diff, \
                    interpolation='nearest', \
                    cmap=colormap, \
                    origin='lower',\
                    vmax= maxscale, \
                    vmin= minscale)
   plt.colorbar(im, use_gridspec=True)
   ax.text( 3.,3., \
            'mean=%.2f\nstddev=%.2f' %  (numpy.mean( m_diff.compressed() ), numpy.std( m_diff.compressed()) ),\
            fontsize=14, color='blue', weight='bold') 
  
   #Scaling 
   ax = plt.subplot2grid( (rows, cols), (1,1), rowspan=1, colspan=1)
   ax.set_title( 'atoms-$\eta$*noatoms  $\eta=%.2f$' % eta)
   colormap = matplotlib.cm.spectral
   m_diffscale = numpy.ma.MaskedArray( diffscale, mask=eigenmask) 
   im = ax.imshow( m_diffscale, \
                    interpolation='nearest', \
                    cmap=colormap, \
                    origin='lower',\
                    vmax= maxscale, \
                    vmin= minscale)
   plt.colorbar(im, use_gridspec=True)
   ax.text( 3.,3., \
            'mean=%.2f\nstddev=%.2f' %  (numpy.mean( m_diffscale.compressed() ), numpy.std( m_diffscale.compressed()) ),\
            fontsize=14, color='blue', weight='bold') 
   

   #Eigenface
   ax = plt.subplot2grid( (rows, cols), (1,2), rowspan=1, colspan=1)
   ax.set_title( r'eigenface clean')
   colormap = matplotlib.cm.spectral
   m_eigen= numpy.ma.MaskedArray( eigen, mask=eigenmask) 
   im = ax.imshow( m_eigen, \
                    interpolation='nearest', \
                    cmap=colormap, \
                    origin='lower',\
                    vmax= maxscale, \
                    vmin= minscale)
   plt.colorbar(im, use_gridspec=True)
   ax.text( 3.,3., \
            'mean=%.2f\nstddev=%.2f' %  (numpy.mean( m_eigen.compressed() ), numpy.std( m_eigen.compressed()) ),\
            fontsize=14, color='blue', weight='bold')

 

   ####### PLOT RESULTING ROIS #######

   roi_diff  = numpy.ma.MaskedArray( diff, mask =  1-mask)
   roi_diffscale = numpy.ma.MaskedArray( diffscale, mask = 1-mask) 
   roi_eigen = numpy.ma.MaskedArray( eigen, mask = 1-mask)

   #Do the counting over the pixels of interest
   bragg_diff=[]
   bragg_diffscale=[]
   bragg_eigen=[]
   troi_diff      = numpy.transpose( roi_diff )
   troi_diffscale = numpy.transpose( roi_diffscale )
   troi_eigen     = numpy.transpose( roi_eigen )
   for px in pixels:
     #print px, ' = ', troi_eigen[px]
     bragg_diff.append( troi_diff[ px ] ) 
     bragg_diffscale.append( troi_diffscale[ px ] ) 
     bragg_eigen.append( troi_eigen[ px ] ) 
   bragg_diff = numpy.array(bragg_diff)
   bragg_diffscale = numpy.array(bragg_diffscale)
   bragg_eigen = numpy.array(bragg_eigen) 
   
   #Standard deviation outside mask
   stdev_out_diff = numpy.std( m_diff.compressed() ) 
   stdev_out_diffscale = numpy.std( m_diffscale.compressed() ) 
   stdev_out_eigen = numpy.std( m_eigen.compressed() ) 
   #Mean outside mask
   mean_out_diff = numpy.mean( m_diff.compressed() ) 
   mean_out_diffscale = numpy.mean( m_diffscale.compressed() ) 
   mean_out_eigen = numpy.mean( m_eigen.compressed() ) 
   #Signal
   signal_diff = bragg_diff.sum() - bragg_diff.size * mean_out_diff
   error_diff  = numpy.sqrt( bragg_diff.size ) * stdev_out_diff
 
   #signal_diffscale = bragg_diffscale.sum() - bragg_diffscale.size * mean_out_diffscale
   signal_diffscale = bragg_diffscale.sum() 
   error_diffscale  = numpy.sqrt( bragg_diffscale.size ) * stdev_out_diffscale 

   #signal_eigen = bragg_eigen.sum() - bragg_eigen.size * mean_out_eigen
   signal_eigen = bragg_eigen.sum() 
   error_eigen  = numpy.sqrt( bragg_eigen.size ) * stdev_out_eigen


   #Make the plots 
   xb = numpy.where( mask.sum(axis=0) != 0 )[0]
   yb = numpy.where( mask.sum(axis=1) != 0 )[0]
   minscale = min( roi_diff.min(), roi_diffscale.min(), roi_eigen.min() )
   maxscale = max( roi_diff.max(), roi_diffscale.max(), roi_eigen.max() )*1.01

   boxeslw = 0.
   boxeslw = 1.5
   boxeslw = 0.4
   boxesal = 0.6
   boxescol = 'black'

   def plotROI(title, ij, roi, signal, error):
     if True:
       ax = plt.subplot2grid( (rows, cols), ij, rowspan=1, colspan=1)
       ax.set_title( title )
       colormap = matplotlib.cm.jet
       im = ax.imshow( roi, \
                        interpolation='nearest', \
                        cmap=colormap, \
                        origin='lower',\
                        vmax= maxscale, \
                        vmin= minscale)
       ax.set_xlim( xb[0]-0.5, xb[-1]+0.5)
       ax.set_ylim( yb[0]-0.5, yb[-1]+0.5)
       plt.colorbar(im, use_gridspec=True)
       ax.text( 0.+xb[0],0.+yb[0], \
            'sum=%.2f\n+/-%.2f' %  (signal,error) ,\
            fontsize=14, color='white', weight='bold') 
       for px in pixels:
           rect = matplotlib.patches.Rectangle( \
                  (px[0]-0.5, px[1]-0.5), 1, 1, \
                  fill=False, ec=boxescol, lw=boxeslw, alpha=boxesal) 
           ax.add_patch(rect) 
       return ax
     else:
       # This was an attempt to make a surface plot, does not look good
       ax = plt.subplot2grid( (rows, cols), ij, rowspan=1, colspan=1, projection='3d')
       ax.set_title( title )
       colormap = matplotlib.cm.jet
       X = numpy.arange( 0, roi.shape[0])
       Y = numpy.arange( 0, roi.shape[0])
       X,Y = numpy.meshgrid(X, Y)
       surf = ax.plot_surface( X, Y, roi, cmap=colormap)
       ax.set_xlim( xb[0]-0.5, xb[-1]+0.5)
       ax.set_ylim( yb[0]-0.5, yb[-1]+0.5)
       plt.colorbar( surf, use_gridspec=True)
       return ax 

   ax = plotROI( r'atoms-noatoms ROI', (2,0), roi_diff, signal_diff, error_diff) 
   ax = plotROI( r'atoms-$\eta$*noatoms ROI', (2,1), roi_diffscale, signal_diffscale, error_diffscale )
   ax = plotROI( r'eigenface clean ROI', (2,2), roi_eigen, signal_eigen, error_eigen )

   # Check discrepancies between the three methods
   # Warn user if more than 10% 
   def relerr( sig1, sig2):
       return 2.*numpy.abs(sig1-sig2) / (sig1+sig2)

   disc_DS  = relerr(signal_diff, signal_diffscale) 
   disc_DE      = relerr(signal_diff, signal_eigen)
   disc_SE = relerr(signal_diffscale, signal_eigen)  

   warn_threshold = 0.1
   if disc_DS > warn_threshold:
       print "Warning : Discrepancy between SUBTRACT and SCALE = %.2f%%"% ( disc_DS*100.)
   if disc_DE > warn_threshold:
       print "Warning : Discrepancy between SUBTRACT and EIGEN = %.2f%%"% ( disc_DE*100.)

   if disc_SE > warn_threshold:
	print "Warning : Discrepancy between    SCALE and EIGEN = %.2f%%"% ( disc_SE*100.)

   

   plt.tight_layout()

   if not os.path.exists(savepath):
     os.makedirs(savepath)
   
   plt.savefig( savepath + fileprefix + '_eigenBragg.png', dpi=120 )
   plt.close('all') 

   results = {}
   results['roi'] = roi_eigen
   results['masked'] = m_eigen

   results['signal'] = signal_eigen
   results['signalD'] = signal_diff
   results['signalS'] = signal_diffscale

   results['disc_DS'] = disc_DS
   results['disc_DE'] = disc_DE
   results['disc_SE'] = disc_SE

   results['error']  = error_eigen
   results['stdev_out'] = stdev_out_eigen
   results['mean_out'] = mean_out_eigen  

   return results
# --------------------- MAIN CODE  --------------------#


if __name__ == "__main__":
  parser = argparse.ArgumentParser('braggEigen.py')

  parser.add_argument('shot', \
         action="store", type=str, \
         help='shot number to clean up')
  args = parser.parse_args()

  #ANDOR2 ROI for 8x8 BINNING
  roi=[30,26,8,8]
  #ANDOR2 DISCARD for 8x8 BINNING
  discard=[5,5,54,54] 

  #Needs shot and roi to perform analysis:
  atoms     = pyfits.open( args.shot + 'atoms_andor2.fits' )[0].data[0]
  noatoms   = pyfits.open( args.shot + 'noatoms_andor2.fits' )[0].data[0]

  braggpixels=[ (33,30),(33,29), (33,31), (32,29), (32,30), (32,31), (34,29), (34,30), (34,31) ] 

  # Function call is eigenclean_Bragg( atoms, noatoms, roi, shot, fileprefix, camera):
  # if atoms,noatoms are not associated with a single shot then shot can be = None
  
  camera = 'andor2' 
  results = eigenclean_Bragg( atoms, noatoms, discard, roi, pixels, args.shot, args.shot, camera)
  print results['signal']
  print results['error']
  





