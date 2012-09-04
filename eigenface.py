#!/usr/bin/python
import argparse
import glob
import pyfits
import os
import numpy
#from numppy import
#import scipy
from scipy import linalg, mat

import sys
sys.path.append('/home/pmd/py')
import fits2png

import cPickle
import time

verbose = False

############## FUNCTIONS ###############

def GetBackgrounds(Nbgs,shot):
  """This function looks in the current directory and gets the last
     Nbgs available backgrounds.  It checks that all backgrounds have
     the same shape.  It returns a list with the backgrounds, a 
     list with the corresponding shot number, and the shape of 
     the backgrounds. 
     """
  backgrounds = []
  bgshots = []
  shape = None

  pwd = os.getcwd()
  atoms      = glob.glob( pwd + '/????atoms.fits')[-Nbgs:]
  
  thisone = pwd + '/' + shot + 'atoms.fits'
  if thisone in atoms:
    atoms.remove( thisone )
  atoms.append( thisone )
  atoms.sort()
  for img in atoms: 
    try:
      shot = os.path.basename(img).rsplit('atoms')[0]
      atoms_img = pyfits.open( img )[0].data[0]
      noatoms_img = pyfits.open( pwd + '/' + shot + 'noatoms.fits')[0].data[0]
      atomsref_img = pyfits.open( pwd + '/' + shot + 'atomsref.fits')[0].data[0]
      noatomsref_img = pyfits.open( pwd + '/' + shot + 'noatomsref.fits')[0].data[0]
      if shape:
        if atoms_img.shape != shape:
          print "error: found image of different shape %s" % img
          exit(1)
      else:
        shape = atoms_img.shape
      backgrounds.append( (noatoms_img - noatomsref_img) )
      bgshots.append( shot )
    except:
      print "error opening .fits image : %s" % img
      exit(1)
  return backgrounds, bgshots, shape


def Bjk( roi, mask, backgrounds, bgshots ):
  """ This function calculates the entries of the Bjk matrix 
      which is used in the calculation of the coefficients
      that produce the eigenclean background.  The mask must
      be provided and it should cover any part of the frame
      where there are atoms.
      This function returns the Bjk matrix and also the 
      normalization factor used to keep the entries of the Bjk
      matrix in the range (0,1). 
      The same normalization factor must be used to calculate
      the entries of Yj,  see below.
   
      To improve the performance, any Bjk elements calculated
      here are stored in a dictionary that is dumped into a 
      pickle file at the end of this function.  Elementes that
      have been calculated before can be retrieved from the 
      dictionary. 
  """
  start = time.time()
  lb = len(backgrounds)
  bjk = numpy.zeros( (lb,lb) ) 
  fpck = 'Bjk_' + str(roi) + '.pickle'
  try:
    Bjk_pickle = open(fpck,'rb')
    Bjk_dict = cPickle.load( Bjk_pickle )
    Bjk_pickle.close()
  except: 
    Bjk_dict = {}
  for j in range( lb ):
    for k in range( lb ):
      try:
        bjk[j,k] = Bjk_dict[ bgshots[j],bgshots[k] ]
      except:
        bjk[j,k] = numpy.sum(mask * backgrounds[j] * backgrounds[k] )
        Bjk_dict[bgshots[j],bgshots[k]] = bjk[j,k]
  Bjk_pickle = open(fpck,'wb')
  cPickle.dump( Bjk_dict, Bjk_pickle)
  bjk_max = numpy.max( bjk)
  bjk = bjk / bjk_max
  end = time.time()
  if ( verbose ):
    print "Bjk evaluation time = %.2f seconds" % (end - start)
  return numpy.matrix( bjk ), bjk_max


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
    yj[j] = numpy.sum( mask * backgrounds[j] * image ) / normfactor
  end = time.time()
  if ( verbose ):
    print "Yj evaluation time = %.2f seconds" % (end - start)
  return numpy.matrix( yj ).T


def makemask( shape, roi):
  """ This function makes the mask array given a shape
      and the region of interest where the atoms are, roi. 
      The roi is a list of four integers like this, such that
      the atoms part of the image is determined by 
        [ roi[0] : roi[0] + roi[2],  roi[1] : roi[1] + roi[3]
  """
  a = numpy.ones(shape)
  a[ roi[0]:roi[0]+roi[2], roi[1]:roi[1]+roi[3] ] = 0.
  #print a
  return a


def Ax( shot ):
  atomsf = os.getcwd() + '/' + shot + 'atoms.fits'
  atomsreff = os.getcwd() + '/' + shot + 'atomsref.fits'
  atoms_img = pyfits.open( atomsf )[0].data[0]
  atomsref_img = pyfits.open( atomsreff )[0].data[0]
  return atoms_img - atomsref_img, atomsf

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
  bgs,bgshots,shape = GetBackgrounds(Nbgs,shot)
  if ( verbose ):
    print "GetBackgrounds process time = %.2f seconds" % (time.time()-t0)

  mask = makemask( shape, roi)
  B, normfactor =  Bjk( roi, mask, bgs , bgshots)

  if ( verbose ):
    print "# of backgrounds = %d" % len(bgs)
    print "pixels per background = %s" % str(bgs[0].shape)
    print "mask roi = %s" % str(roi)
    print "shape of B matrix = %s" % str(B.shape)

  image, atomsf = Ax(shot)
  y = Yj( mask, bgs, image, normfactor)  

  if ( verbose ):
    print "shape of y vector = %s" % str(y.shape)

  t0 = time.time()
  c, resid, rank, sigma = linalg.lstsq( B, y)
  if ( verbose ):
    print "Coeffs evaluation time = %.2f seconds" % (time.time()-t0)
    print "maximum residual = %.3e" % numpy.max(y-B*c)

  if ( verbose ):
    print "\neigenclean background:\n"  
  for i in range( c.shape[0]  ):
    out =  "% 2.3f * '%s'" % ( c[i], bgshots[i] ) 
    if bgshots[i] == shot :
      out = out + ' <--'
    if ( verbose ):
      print out

  t0 = time.time()
  nb = NewBackground( c, bgs) 
  if ( verbose ):
    print "New background evaluation time = %.2f seconds" % (time.time()-t0)
  t0 = time.time()
  numpy.savetxt( shot + '_eigenclean.ascii' , nb, fmt='%.6e', delimiter='\t')
  if ( verbose ):
    print "New background save to disk time = %.2f seconds" % (time.time()-t0)

  fits2png.makepng( atomsf, 'ABS', 140, prefix = '_noclean')
  fits2png.makepng( atomsf, 'ABS', 140, bg =nb,  prefix = '_clean')

############## USAGE EXAMPLE ###############

if __name__ == "__main__":
 
  parser = argparse.ArgumentParser('eigenface.py')
  parser.add_argument('SHOT', action="store", type=str, help='shotnumber to clean up with eigenface')
  parser.add_argument('ROI', action="store", type=str, help='X0,Y0,XW,YW region of interest') 
  parser.add_argument('NBGS', action="store", type=int, help='maximum number of backgrounds to be used') 
  
  args = parser.parse_args()
  X0 = float(args.ROI.split(',')[0])
  Y0 = float(args.ROI.split(',')[1])
  XW = float(args.ROI.split(',')[2])
  YW = float(args.ROI.split(',')[3])
 
#  print args.SHOT
#  print X0,Y0,XW,YW
 
  EigenClean( args.SHOT, [X0,Y0,XW,YW], Nbgs = args.NBGS)

