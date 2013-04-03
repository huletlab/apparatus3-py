#!/usr/bin/python

import sys 
import pyfits
import wx

sys.path.append('/lab/software/apparatus3/bin/py')
import falsecolor


def makepng( fitsfile, operation, dpi, bg = None, prefix = None):
  
   shot = fitsfile.split('atoms')[0]

   print "Inside makepng:"
   print fitsfile

   if 'andor2' in fitsfile:
     atoms     = pyfits.open( shot + 'atoms_andor2.fits')[0].data[0]
     noatoms   = pyfits.open( shot + 'noatoms_andor2.fits')[0].data[0]
     atomsref  = pyfits.open( shot + 'atomsref_andor2.fits')[0].data[0]
     noatomsref= pyfits.open( shot + 'noatomsref_andor2.fits')[0].data[0]
   else:
     atoms     = pyfits.open( shot + 'atoms.fits')[0].data[0]
     noatoms   = pyfits.open( shot + 'noatoms.fits')[0].data[0]
     atomsref  = pyfits.open( shot + 'atomsref.fits')[0].data[0]
     noatomsref= pyfits.open( shot + 'noatomsref.fits')[0].data[0]
   
   if operation == 'ABS':
      if bg == None:
        out = (atoms - atomsref) / (noatoms - noatomsref)
      else:
        out = (atoms - atomsref) / bg
   elif operation == 'PHC':
      if bg == None:
        out = (atoms - atomsref) - (noatoms - noatomsref) 
      else:
        out = (atoms - atomsref) - bg
   else:
      print " -->  Operation is not ABS or PHC.  Program will exit"
      exit(1) 
  
   if prefix == None: 
     label = shot
   else:
     label = shot + prefix 
  
   return falsecolor.savepng( out, out.min(), out.max(), falsecolor.my_rainbow, label, dpi)

if __name__ == "__main__":
   #parameter indeces
   FITSFILE=1
   OPERATION=2

   if not (len(sys.argv) == OPERATION+1 ):
       print "  fits2png.py:"
       print ""
       print "  Looks at the four fits images that match FITSFILE and produces a"
       print "  png image that is the result of applying OPERATION on the set of"
       print "  four images."
       print ""
       print ""
       print "  usage:  fits2png.py [FITSFILE] [OPERATION]" 
       print "" 
       print "  FITSFILE has to be: ####atoms.fits"
       print ""
       print "  OPERATION has to be: ABS (absorption) or PHC (phase contrast)"  
       print ""
       print "  If OPERATION == ABS the png image will be (1-3)/(2-4)"
       print "  If OPERATION == PHC the png image will be (1-3)-(2-4)"
       print "" 
       print "  where 1=atoms, 2=noatoms, 3=atomsreference, 4=noatomsreference"
       print ""
       print "  Examples:"
       print "  fits2png.py 6450atoms.fits ABS"
       print "  fits2png.py 6451atoms.fits PHC"    
       exit(1)

   makepng( sys.argv[FITSFILE], sys.argv[OPERATION], 75)
	    


