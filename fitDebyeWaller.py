
import numpy
from scipy import optimize

def fitDW( data, p0 , datarange = None ):
  """
  p = [ amplitude, harmonic osc. length ]
  """
  if datarange != None:

        inside = numpy.ma.masked_outside( data[:,0], datarange[0], datarange[1])
        index = []
        for i in range(len(inside.mask)):
           if inside.mask[i] == False:
             index.append(i)
        data =  data [ index, :]

  fitfunc = lambda p, x: p[0] * numpy.exp( - 1./2. * p[1]**2 / numpy.sqrt(x) ) 
  errfunc = lambda p, x, y: fitfunc(p,x) - y 
  p, cov, infodict, errmsg, success = optimize.leastsq(errfunc, p0[:], args=(data[:,0], data[:,1]), full_output=1)
  fit = fitfunc(p,data[:,0])
  fitdata = numpy.transpose( numpy.array( [data[:,0], fit] ))
  return p, fitdata

def evalDW( p0, xpoints):
  """
  p = [ amplitude, harmonic osc. length ]
  """
  fitfunc = lambda p, x: p[0] * numpy.exp( - 1./2. * p[1]**2 / numpy.sqrt( x) ) 
  ypoints = fitfunc(p0,xpoints)
  return numpy.transpose( numpy.array( [xpoints, ypoints] ) )


