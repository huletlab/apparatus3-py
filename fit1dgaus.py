
import numpy
from scipy import optimize

def fit1dgaus( data, p0 ):
  """
  p = [ amplitude, center, 1/e^2 radius, offset]
  """
  fitfunc = lambda p, x: p[0]*numpy.exp( -2* ((x-p[1])/p[2])**2 ) + p[3]
  errfunc = lambda p, x, y: fitfunc(p,x) - y 
  p, cov, infodict, errmsg, success = optimize.leastsq(errfunc, p0[:], args=(data[:,0], data[:,1]), full_output=1)
  fit = fitfunc(p,data[:,0])
  fitdata = numpy.transpose( numpy.array( [data[:,0], fit] ))
  return p, fitdata

def eval1dgaus( p0, xpoints):
  """
  p = [ amplitude, center, 1/e^2 radius, offset]
  """
  fitfunc = lambda p, x: p[0]*numpy.exp( -2* ((x-p[1])/p[2])**2 ) + p[3]
  ypoints = fitfunc(p0,xpoints)
  return numpy.transpose( numpy.array( [xpoints, ypoints] ) )


