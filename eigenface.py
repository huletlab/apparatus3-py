#!/usr/bin/python
import argparse
import glob
import pyfits
import os
import numpy
#from numppy import
import scipy
from scipy import linalg, mat
from configobj import ConfigObj

import matplotlib.pyplot as plt
import matplotlib

import sys
sys.path.append('/lab/software/apparatus3/bin/py')
sys.path.append('/lab/software/apparatus3/py')

import cPickle
import time

verbose = True
very_verbose = False

############## FUNCTIONS ###############

def getnoatoms( shot ):
  pwd = os.getcwd()
  try:
    atoms = pyfits.open( pwd + '/' + shot + 'noatoms.fits')[0].data[0] 
    atomsref = pyfits.open( pwd + '/' + shot + 'noatomsref.fits')[0].data[0]
    return atoms-atomsref
  except:
    print "Error loading noatoms fits files for shot #%04d" % shot
    exit(1) 

def getatoms( shot ):
  pwd = os.getcwd()
  try:
    atoms = pyfits.open( pwd + '/' + shot + 'atoms.fits')[0].data[0]
    atomsref = pyfits.open( pwd + '/' + shot + 'atomsref.fits')[0].data[0]
    return atoms-atomsref
  except:
    print "Error loading atoms fits files for shot #%04d" % shot
    exit(1) 

 
def GetBackgrounds(Nbgs,thisshot):
  """This function looks in the current directory and gets the closest
     Nbgs available backgrounds.  It checks that all backgrounds have
     the same shape.  It returns a list with the backgrounds, a 
     list with the corresponding shot number, and the shape of 
     the backgrounds. 
     """
  backgrounds = []
  bgshots = []
  shape = None

  #This works for shots taken by andor1
  fitskey = '.fits'
   
  if thisshot != None:
    noatoms_img = getnoatoms( thisshot ) 
    backgrounds.append( noatoms_img  )
    bgshots.append( thisshot )

  if Nbgs==0:
    return backgrounds, bgshots, shape

  atoms      = glob.glob( os.getcwd() + '/????atoms' + fitskey )
  shots = [ os.path.basename( a ).split('atoms')[0]  for a in atoms ]  
  #print "This is shot #", thisshot

  # Here, need to select the shots that are closest to thisshot and 
  # that match some basic report keys  
  # For this purpose, first sort the list by proximity to thisshot 
  keyfun = lambda x : (int(x) - int(thisshot))**2 
  shots = sorted( shots, key = keyfun )

  # Then start looking for the desired keys in 
  keys = [ ('ANDOR','exp') , ('ANDOR','phc'), ('ANDOR','fluor') ]
  vals = []
  inifile = "report" + thisshot + ".INI"
  report = ConfigObj(inifile)
  for sec,key in keys:
      try:
        vals.append( report[ sec ][ key ] ) 
      except:
        print "Error finding sec:key = %s:%s in:" % (sec, key)
        print inifile
        exit(1)

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
  
  for img in eigenshots:
    if img == thisshot:
      continue 
    try:
      noatoms_img = getnoatoms( img ) 
      if shape:
        if noatoms_img.shape != shape:
          #print "error: found image of different shape %s" % img
          continue
      else:
        shape = noatoms_img.shape
      backgrounds.append( noatoms_img  )
      bgshots.append( img )
    except:
      print "error opening image : %s" % img
      exit(1)
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


def makemask( shape, roi):
  """ This function makes the mask array given a shape
      and the region of interest where the atoms are, roi. 
      The roi is a list of four integers like this, such that
      the atoms part of the image is determined by 
        [ roi[0] : roi[0] + roi[2],  roi[1] : roi[1] + roi[3]
  """
  a = numpy.zeros(shape)
  a[ roi[0]:roi[0]+roi[2], roi[1]:roi[1]+roi[3] ] = 1.
  #print a
  return a


def Ax( shot ):
  atomsf = os.getcwd() + '/' + shot + 'atoms.fits'
  return getatoms(shot), atomsf

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


############## EIGENCLEAN ALGORITHM ###############


def EigenClean( shot, roi,  Nbgs=40):
  t0 = time.time()
  start = t0
  bgs,bgshots,shape = GetBackgrounds(Nbgs,shot)

  if ( very_verbose ):
    print "GetBackgrounds process time = %.2f seconds" % (time.time()-t0)

  eigenmask = makemask( shape, roi)
  B, normfactor =  Bjk( eigenmask, bgs )

  if ( verbose or True ):
    print "Bjk process time = %.2f seconds" % (time.time()-t0)
  if ( verbose ):
    print "\t# of backgrounds = %d" % len(bgs)
    print "\tpixels per background = %s" % str(bgs[0].shape)
    print "\tmask roi = %s" % str(roi)
    print "\tshape of B matrix = %s" % str(B.shape)

  image, atomsf = Ax(shot)
  y = Yj( eigenmask, bgs, image, normfactor)  
  if ( verbose or True ):
    print "Yj process time = %.2f seconds" % (time.time()-t0)

  if ( very_verbose ):
    print "shape of y vector = %s" % str(y.shape)

  t0 = time.time()
  c, resid, rank, sigma = linalg.lstsq( B, y)
  if ( verbose ):
    print "Coeffs evaluation time = %.2f seconds" % (time.time()-t0)
    print "maximum residual = %.3e" % numpy.max(y-B*c)

  if ( very_verbose ):
    print "\neigenclean background:\n"  
    for i in range( c.shape[0]  ):
      out =  "% 2.3f * '%s'" % ( c[i], bgshots[i] ) 
      if bgshots[i] == shot :
        out = out + ' <--'
      print out

  t0 = time.time()
  newbg = NewBackground( c, bgs) 
  if ( verbose ):
    print "New background evaluation time = %.2f seconds" % (time.time()-t0)
  t0 = time.time()
  numpy.savetxt( shot + '_eigenclean.ascii' , newbg, fmt='%.6e', delimiter='\t')
  if ( verbose ):
    print "New background save to disk time = %.2f seconds" % (time.time()-t0)
 
  end = t0
  print "Time spent on eigenface : %.2f seconds\n" % (end-start)


  ####### MAKE PLOTS ILLUSTRATING EIGENFACE RESULTS #######
  atoms = image
  noatoms = getnoatoms(shot) 
  eigen = atoms - newbg

  fig = plt.figure(figsize=(12.,8.))

  rows = 2
  cols = 3
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

  plt.tight_layout()

  savepath = 'andor1eigen/' 
  if not os.path.exists(savepath):
    os.makedirs(savepath)
   
  plt.savefig( savepath +  shot + '_eigen.png', dpi=120 )
  plt.close('all') 
 



############## USAGE EXAMPLE ###############

if __name__ == "__main__":
 
  parser = argparse.ArgumentParser('eigenface.py')
  parser.add_argument('SHOT', action="store", 
                      type=str, 
                      help='shotnumber to clean up with eigenface')

  parser.add_argument('ROI', action="store", 
                      type=str, 
                      help='X0,Y0,XW,YW region of interest') 

  parser.add_argument('NBGS', action="store", 
                      type=int, 
                      help='maximum number of backgrounds to be used') 
  
  args = parser.parse_args()
  X0 = float(args.ROI.split(',')[0])
  Y0 = float(args.ROI.split(',')[1])
  XW = float(args.ROI.split(',')[2])
  YW = float(args.ROI.split(',')[3])
 
#  print args.SHOT
#  print X0,Y0,XW,YW
 
  #EigenClean( args.SHOT, [X0,Y0,XW,YW], Nbgs = args.NBGS)

  #Override the number of bgs set by C++
  EigenClean( args.SHOT, [Y0,X0,YW,XW], Nbgs = 100)

