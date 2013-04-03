#!/usr/bin/python

import sys 
import pyfits

sys.path.append('/lab/software/apparatus3/bin/py')

import fit2dlibrary
import gaussfitter

if __name__ == "__main__":
   #parameter indeces
   FITSFILE=1

   image =  pyfits.open( sys.argv[FITSFILE] )[0].data[0]
   out = image

   p0 = 100.
   p1 = 256.
   p2 = 256.
   p3 = 15.
   p4 = 15.0
   p5 = 2.0
   p6 = 0.  
   
   pFit, error = fit2dlibrary.fit_function(  p0, image, fit2dlibrary.fitdict['Gaussian2Dphi'].function ) 

   print pFit

   pstart = [patchcounts/box/box,braggsig, row, col, 5., 5., 0.] 
   p, fitimg =  gaussfitter.gaussfit( out, params=pstart, returnfitimage=True)
   # p = [height, amplitude, x, y, width_x, width_y, rotation]

   row = p[2]
   col = p[3] 

   print "Sum of counts = %f" % image.sum() 
 
   pngprefix =  sys.argv[FITSFILE] + '_fits'
   falsecolor.inspecpng( [image, fitimg], row, col, out.min(), out.max(), \
                         falsecolor.my_grayscale, pngprefix, 100, origin = 'upper' , step=False, scale=10) 
  

	    


