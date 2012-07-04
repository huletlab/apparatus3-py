#!/lab/software/epd-7.0-1-rh5-x86/bin/python

import sys 
import pyfits
import wx
import numpy

sys.path.append('/lab/software/apparatus3/bin/py')
import falsecolor


def makepng( atomsfile, operation, dpi):
   shot = atomsfile.split('atoms')[0]
   atoms     = numpy.loadtxt( shot + 'atoms.manta')
   noatoms   = numpy.loadtxt( shot + 'noatoms.manta')
   atomsref  = numpy.loadtxt( shot + 'atomsref.manta')
   noatomsref= numpy.loadtxt( shot + 'noatomsref.manta') 
   
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
   ATOMSFILE=1
   OPERATION=2

   if not (len(sys.argv) == OPERATION+1 ):
       print "  manta2png.py:"
       print ""
       print "  Looks at the four .manta images that match ATOMSFILE and produces a"
       print "  png image that is the result of applying OPERATION on the set of"
       print "  four images."
       print ""
       print ""
       print "  usage:  manta2png.py [ATOMSFILE] [OPERATION]" 
       print "" 
       print "  ATOMSFILE has to be: ####atoms.manta"
       print ""
       print "  OPERATION has to be: ABS (absorption) or PHC (phase contrast)"  
       print ""
       print "  If OPERATION == ABS the png image will be (1-3)/(2-4)"
       print "  If OPERATION == PHC the png image will be (1-3)-(2-4)"
       print "" 
       print "  where 1=atoms, 2=noatoms, 3=atomsreference, 4=noatomsreference"
       print ""
       print "  Examples:"
       print "  manta2png.py 6450atoms.manta ABS"
       print "  manta2png.py 6451atoms.manta PHC"    
       exit(1)

   makepng( sys.argv[ATOMSFILE], sys.argv[OPERATION], 75)
	    


