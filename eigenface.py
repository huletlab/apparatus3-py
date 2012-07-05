import glob
import pyfits
import os
import numpy
import scipy
from scipy import linalg

import sys
sys.path.append('/home/pmd/py')
import fits2png


############## FUNCTIONS ###############

def GetBackgrounds():
  """This function looks in the current directory and gets all 
     available backgrounds.  It checks that all backgrounds have
     the same shape.  It returns a list with the backgrounds, a 
     list with the corresponding shot number, and the shape of 
     the backgrounds. 
     """
  backgrounds = []
  bgshots = []
  shape = None

  atoms      = glob.glob( os.getcwd() + '/????atoms.fits')
  atoms.sort()
  for img in atoms: 
    try:
      pwd = os.getcwd()
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


def Bjk( mask, backgrounds ):
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
  """
  lb = len(backgrounds)
  bjk = numpy.zeros( (lb,lb) ) 
  for j in range( lb ):
    for k in range( lb ):
      bjk[j,k] = numpy.sum(mask * backgrounds[j] * backgrounds[k] )
  bjk_max = numpy.max( bjk)
  bjk = bjk / bjk_max
  return scipy.mat( bjk ), bjk_max


def Yj( mask, backgrounds, image, normfactor):
  """ This function calculates the entries of the Yj vector
      which is used in the calculation of the coefficients
      that produce the eigenclean background.  The mask 
      must be provided. 
      The normalization factor used to normalize Bjk must
      also be provided.  
  """
  lb = len(backgrounds)
  yj = numpy.zeros( lb ) 
  for j in range( lb ):
    yj[j] = numpy.sum( mask * backgrounds[j] * image ) / normfactor
  return scipy.mat( yj ).T


def makemask( shape, roi):
  """ This function makes the mask array given a shape
      and the region of interest where the atoms are, roi. 
      The roi is a list of four integers like this, such that
      the atoms part of the image is determined by 
        [ roi[0] : roi[0] + roi[2],  roi[1] : roi[1] + roi[3]
  """
  a = numpy.ones(shape)
  a[ roi[0]:roi[0]+roi[2], roi[1]:roi[1]+roi[3] ] = 0.
  print a
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

bgs,bgshots,shape = GetBackgrounds()
mask = makemask( shape, [200,200,200,250] )
B, normfactor =  Bjk( mask, bgs)

print "# of backgrounds = %d" % len(bgs)
print "pixels per background = %s" % str(bgs[0].shape)
print "shape of B matrix = %s" % str(B.shape)

image, atomsf = Ax('0188')
y = Yj( mask, bgs, image, normfactor)  

print "shape of y vector = %s" % str(y.shape)

c, resid, rank, sigma = linalg.lstsq( B, y)

print  y-B*c
print c
print bgshots

nb = NewBackground( c, bgs) 

fits2png.makepng( atomsf, 'ABS', 140, prefix = '_noclean')
fits2png.makepng( atomsf, 'ABS', 140, bg =nb,  prefix = '_clean')

