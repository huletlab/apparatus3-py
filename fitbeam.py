
import numpy
from scipy import optimize

def fitbeam( data, p0 , wavelength):
  """
  p = [ waist, waist_location]
  waist is in microns, waist_location is in mm
  wavelength is in microns 
  """
  fitfunc = lambda p, x: p[0]*numpy.sqrt( 1 + numpy.power ( 1000 * (x-p[1]) / (numpy.pi * p[0]*p[0]/wavelength)  ,2))
  errfunc = lambda p, x, y: fitfunc(p,x) - y 
  p, cov, infodict, errmsg, success = optimize.leastsq(errfunc, p0[:], args=(data[:,0], data[:,1]), full_output=1)
  fit = fitfunc(p,data[:,0])
  fitdata = numpy.transpose( numpy.array( [data[:,0], fit] ))
  return p, fitdata

def evalbeam( p0, wavelength, xpoints):
  """
  p = [ waist, waist_location]
  waist is in microns, waist_location is in mm
  wavelength is in microns 
  """
  fitfunc = lambda p, x: p[0]*numpy.sqrt( 1 + numpy.power ( 1000 * (x-p[1]) / (numpy.pi * p[0]*p[0]/wavelength)  ,2))
  ypoints = fitfunc(p0,xpoints)
  return numpy.transpose( numpy.array( [xpoints, ypoints] ) )


