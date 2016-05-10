#!/usr/bin/python
import numpy
from scipy import optimize
import math
import matplotlib.pyplot as plt
import argparse
#from scipy.special import erf

#def fitbeamFromFile(f,save=0):
#	sName = f.split(".")[0] + ".png"
#	try: 
#		data = numpy.loadtxt(f) 
#		#data = numpy.transpose(numpy.array([[1,2,3,4],[3,4,5,6]]))
#		m = max(data[:,1])
#		x0 = min(data[:,0])
#		x1 = max(data[:,0])
#		#exit(0)
#		p= [m,(x1-x0)/10,(x0+x1)/2,0]
#		p0,fd = fitbeam(data,p)
#		print "wasit is",p0[1],"Unit"
#		if save:
#			pl = plt.figure()
#			ax= pl.add_subplot(111)
#			ax.plot(data[:,0],data[:,1])
#			ax.plot(fd[:,0],fd[:,1])
#			print "Saving file to" + sName
#			plt.savefig(sName)
#		return p0,data,fd
#	except:
#		print "Not a valid file"


def fitGaussianWaist( data,fitlambda=0):
  data = numpy.array(data)
  xdata= data[:,0]
  ydata= data[:,1]
  #erf = numpy.vectorize(math.erf)
  if fitlambda:
	p0 = (ydata.min(),xdata[0],xdata[-1]-xdata[0])
	fitfunc = lambda p, x: p[0]*(1+((x-p[1])/p[2])**2)**0.5
  else:	
	p0 = (ydata.min(),xdata[0])
	fitfunc = lambda p, x: p[0]*(1+((x-p[1])/(numpy.pi*p[0]**2/fitlabmbda))**2)**0.5
  
  errfunc = lambda p, x, y: fitfunc(p,x) - y

  p, cov, infodict, errmsg, success = optimize.leastsq(errfunc, p0[:], args=(data[:,0], data[:,1]), full_output=1)
  print success
  fit = fitfunc(p,data[:,0])
  fitdata = numpy.transpose( numpy.array( [data[:,0], fit] ))
  if fitlambda:
	p0 = numpy.append(p0,numpy.pi*p0[0]**2/fitlambda)
  
  return p, fitdata

#if __name__ == "__main__":
#	parser = argparse.ArgumentParser()
#	parser.add_argument("file_name",help="Data file for the razor blade waist measurement with col1: postion of razor blade and col2: Power ")
#	args = parser.parse_args()
#	fName = args.file_name
#	fitbeamFromFile(fName,1)
