#!/lab/software/epd-7.0-1-rh5-x86/bin/python

import sys 
import pyfits
import wx

sys.path.append('/lab/software/apparatus3/bin/py')
import falsecolor


def makepng( fitsfile, operation, dpi):
   shot = fitsfile.split('atoms')[0]
   atoms     = pyfits.open( shot + 'atoms.fits')[0].data[0]
   noatoms   = pyfits.open( shot + 'noatoms.fits')[0].data[0]
   atomsref  = pyfits.open( shot + 'atomsref.fits')[0].data[0]
   noatomsref= pyfits.open( shot + 'noatomsref.fits')[0].data[0]
   
   if operation == 'ABS':
      out = (atoms - atomsref) / (noatoms - noatomsref)
   elif operation == 'PHC':
      out = (atoms - atomsref) - (noatoms - noatomsref) 
   else:
      print " -->  Operation is not ABS or PHC.  Program will exit"
      exit(1) 
   
   return falsecolor.savepng( out, out.min(), out.max(), falsecolor.my_rainbow, shot, dpi)

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
	    


