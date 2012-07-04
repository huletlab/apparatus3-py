
import numpy
from scipy import optimize

def fit1dlorentz( data, p0 , datarange = None ):
  """
  p = [ amplitude, center, linewidth, offset]
  """
  if datarange != None:

        inside = numpy.ma.masked_outside( data[:,0], datarange[0], datarange[1])
        index = []
        for i in range(len(inside.mask)):
           if inside.mask[i] == False:
             index.append(i)
        data =  data [ index, :]

  fitfunc = lambda p, x: p[0]*( 1 / ( numpy.pi * p[2] * ( 1 + (( x - p[1] ) / p[2])**2 ) ) ) + p[3] 
  errfunc = lambda p, x, y: fitfunc(p,x) - y 
  p, cov, infodict, errmsg, success = optimize.leastsq(errfunc, p0[:], args=(data[:,0], data[:,1]), full_output=1)
  fit = fitfunc(p,data[:,0])
  fitdata = numpy.transpose( numpy.array( [data[:,0], fit] ))
  return p, fitdata

def eval1dlorentz( p0, xpoints):
  """
  p = [ amplitude, center, linewidth, offset]
  """
  fitfunc = lambda p, x: p[0]*( 1 / ( numpy.pi * p[2] * ( 1 + (( x - p[1] ) / p[2])**2 ) ) ) + p[3] 
  ypoints = fitfunc(p0,xpoints)
  return numpy.transpose( numpy.array( [xpoints, ypoints] ) )


