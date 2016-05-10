#!/usr/bin/python
import numpy
from scipy import optimize
import math
import matplotlib.pyplot as plt
import argparse
#from scipy.special import erf

def fitbeamFromFile(f,save=0):
	sName = f.split(".")[0] + ".png"
	try: 
		data = numpy.loadtxt(f) 
		#data = numpy.transpose(numpy.array([[1,2,3,4],[3,4,5,6]]))
		m = max(data[:,1])
		x0 = min(data[:,0])
		x1 = max(data[:,0])
		#exit(0)
		p= [m,(x1-x0)/10,(x0+x1)/2,0]
		p0,fd = fitbeam(data,p)
		print "wasit is",p0[1],"Unit"
		if save:
			pl = plt.figure()
			ax= pl.add_subplot(111)
			ax.plot(data[:,0],data[:,1])
			ax.plot(fd[:,0],fd[:,1])
			print "Saving file to" + sName
			plt.savefig(sName)
		return p0,data,fd
	except:
		print "Not a valid file"


def fitbeam( data, p0):
  """
  p = [ scale,waist, waist_location]
  """
  erf = numpy.vectorize(math.erf)
  fitfunc = lambda p, x: p[0]*(1-erf(2.0**0.5/p[1]*(x-p[2])))+p[3]
#  def fitfunc(p, x):
#	result = []
#	for i in x:
#		result.append(math.erfc(2**0.5/p[1]*i)+p[2])
#	return result
	
  errfunc = lambda p, x, y: fitfunc(p,x) - y

  p, cov, infodict, errmsg, success = optimize.leastsq(errfunc, p0[:], args=(data[:,0], data[:,1]), full_output=1)
  print success
  fit = fitfunc(p,data[:,0])
  fitdata = numpy.transpose( numpy.array( [data[:,0], fit] ))
  return p, fitdata

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("file_name",help="Data file for the razor blade waist measurement with col1: postion of razor blade and col2: Power ")
	args = parser.parse_args()
	fName = args.file_name
	fitbeamFromFile(fName,1)
