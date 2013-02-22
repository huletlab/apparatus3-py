#!/usr/bin/python

import sys
import numpy

import argparse

parser = argparse.ArgumentParser('bfield.py')
parser.add_argument('CURRENT', action='store', type=float, help='magnetic field') 
args = parser.parse_args()


toGauss = 6.87

bfield = args.CURRENT * toGauss


zeeman = numpy.loadtxt('/lab/software/apparatus3/py/2S2Pcombined.dat')

offset = -77.813 - 100.  #Account for the zero field value of the levels
                        #and for the -100 MHz of the imaging AOM

from scipy.interpolate import interp1d

imaging1 = numpy.zeros( (zeeman.shape[0],2))
imaging1[:,0] = zeeman[ :, 0 ]
imaging1[:,1] = zeeman[ :, 1] - zeeman[ :,8] + offset
hfimg1 = interp1d( imaging1[:,0], imaging1[:,1])

imaging2 = numpy.zeros( (zeeman.shape[0],2))
imaging2[:,0] = zeeman[ :, 0 ]
imaging2[:,1] = zeeman[ :, 2] - zeeman[ :,9] + offset
hfimg2 = interp1d( imaging2[:,0], imaging2[:,1])

state1to2 = interp1d( zeeman[:,0], zeeman[:,2] - zeeman[:,1])

state2to3 = interp1d( zeeman[:,0], zeeman[:,3] - zeeman[:,2])

state1to6 = interp1d( zeeman[:,0], zeeman[:,6] - zeeman[:,1])

bfc = 6.6 #bfield correction

print "\n  set point  = %.2f Amps" % args.CURRENT
print   "  conversion = %.2f Gauss/Amp" % toGauss
print   "  field      = %.6f" % bfield

print "\n  IMAGING FREQUENCIES:"
print "    State |1> hfimg = %.1f" % hfimg1(bfield )
print "    State |2> hfimg = %.1f" % hfimg2(bfield )

print "\n  GROUND STATE SPLITTINGS:"
print "  - Diagonalization:"
print "    |1> to |2> = %.4f" % state1to2(bfield)
print "    |2> to |3> = %.4f" % state2to3(bfield)
print "    |1> to |6> = %.4f" % state1to6(bfield)
print "  - Diagonalization plus correction B = B + %.2f:" % bfc
print "    |1> to |2> = %.4f" % state1to2(bfield+ bfc)
print "    |2> to |3> = %.4f" % state2to3(bfield+ bfc)
print "    |1> to |6> = %.4f" % state1to6(bfield+ bfc)

print ""

