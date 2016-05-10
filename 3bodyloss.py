#/lab/software/epd-7.0-1-rh5-x86/bin/python

import sys
import numpy
#import matplotlib.pyplot as plt
#import matplotlib

from scipy import optimize

#This file takes care of analyzing a loss curve 

def fitfunc1( p, x ):
  pp = [ p[0], p[1], 0.5, p[2]]
  return fitfunc( pp,x)
#fitfunc = lambda p, x: p[0] / ( 1 + ( p[1] * x)**p[2] ) ** p[3] 
#fitfunc1 = lambda p, x: p[0] / ( 1 + ( p[1] * x)**0.5 ) ** p[2] 
def fitloss1( data, p0 ):
  errfunc = lambda p, x, y: fitfunc1(p,x) - y
  p, cov, infodict, errmsg, success = optimize.leastsq(errfunc, p0[:], args=(data[:,0]/1e3, data[:,1]/1e6), full_output=1)
  fit = fitfunc1(p,data[:,0]/1.e3)
  fitdata = numpy.transpose( numpy.array( [data[:,0]/1e3, fit] ))
  return p, fitdata
def evaloss1( p0, xpoints):
  ypoints = fitfunc1(p0,xpoints)
  return numpy.transpose( numpy.array( [xpoints, ypoints] ) )

numpy.seterr(over='raise')
numpy.seterr(divide='raise')
numpy.seterr(invalid='raise')
def fitfunc( p, x ):
  if p[1] < 1e-2:
     p[1] = 1e3
  for index,ix in enumerate(x):
    if ix == 0.:
       x[index] = numpy.float64(1e-2)
       #print "Corrected 0.0 wait time"
  try:
    den =  numpy.power( 1 + numpy.power( p[1] * x , p[2] ), p[3])
  except:
    print "Error in fit func denominator. Program will exit" 
    for ix in x:
      print " x = %f" % ix
      print numpy.power( 1 + numpy.power( p[1] * x, p[2] ) , p[3])
    print sys.exc_info()[0]
    print "   p = ", p
    print "   x = ", x
    exit(1)
   
  for index,iden in enumerate(den):
    if iden < 1e-200:
      #print "Corrected small entry in denominator."
      den[index] = 1e-200
  try:
    return p[0]/den
  except: 
    print "Error in fit func. Program will exit"
    print sys.exc_info()[0]
    print "   p = ", p
    print "   x = ", x
    exit(1)
   
#fitfunc = lambda p, x: p[0] / ( 1 + ( p[1] * x)**p[2] ) ** p[3] 
def fitloss( data, p0 ):
  errfunc = lambda p, x, y: fitfunc(p,x) - y
  p, cov, infodict, errmsg, success = optimize.leastsq(errfunc, p0[:], args=(data[:,0]/1e3, data[:,1]/1e6), full_output=1)
  fit = fitfunc(p,data[:,0]/1e3)
  fitdata = numpy.transpose( numpy.array( [data[:,0]/1e3, fit] ))
  return p, fitdata

def evaloss( p0, xpoints):
  ypoints = fitfunc(p0,xpoints)
  return numpy.transpose( numpy.array( [xpoints, ypoints] ) )
def evaloss_rate( p0, xpoints):
  ypoints = -1*(fitfunc(p0,xpoints+1e-4) - fitfunc(p0, xpoints))/ 1e-4/ fitfunc(p0, xpoints)
  return numpy.transpose( numpy.array( [xpoints, ypoints] ) ) 

l1 = numpy.loadtxt('lifetime_77p6.dat', dtype='float64')
l2 = numpy.loadtxt('lifetime_86.dat', dtype='float64')
l3 = numpy.loadtxt('lifetime_90.dat', dtype='float64')
l4 = numpy.loadtxt('lifetime_95.dat', dtype='float64')
l5 = numpy.loadtxt('lifetime_97p66.dat', dtype='float64')
ls= [l1,l2,l3,l4,l5[1:]]

from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
xs=[]
fs=[]
ps=[]
es=[]
xi = [8., 8., 8., 8., 3.]
for i,l in enumerate(ls):
  x = numpy.linspace(0.050, xi[i], 100)
  xs.append(x)
  #print f(1000.)
  p0 = [1.4, 8., 0.4, 2. ]
  try:
    p,fitdat = fitloss( l[:,[1,2]] , p0)
  except:
    print "Failed to fit %d" % i
  ps.append(p)
  print "Fit %d : p = %s" % (i, p)
  es.append( evaloss( p, x) )

  
p0 = [1.4, 8., 2. ]
p,fitdat = fitloss1( l1[:,[1,2]],   p0)
print "Fit Zerocross again : p = %s" % p
ps[0] = [p[0], p[1], 0.5, p[2]]
es[0] = evaloss(ps[0], numpy.linspace(50./1e3, xi[0], 100) )

print "Done with Fits"

import matplotlib
matplotlib.rcdefaults()
from matplotlib import rc
rc('text', usetex=True)
#matplotlib.rcParams['text.latex.preamble']=[r"\usepackage[lf,mathtabular]{MyriadPro}",r'\usepackage{mdsymbol}'] 
import matplotlib.pyplot as plt


fig = plt.figure(figsize=(8,6))
ax1 = fig.add_axes([0.15,0.15,0.75,0.8])


fig2 = plt.figure(figsize=(8,6))
ax2 = fig2.add_axes([0.15,0.15,0.75,0.8])
c = ['green','red','blue','black','deeppink']

def fitcoeff( peakdsq, sqrtrate, p0 ):
  errfunc = lambda p, x, y: p[0]*x - y
  p, cov, infodict, errmsg, success = optimize.leastsq(errfunc, p0[:], args=(peakdsq, sqrtrate), full_output=1)
  fit = p[0] * peakdsq
  fitdata = numpy.transpose( numpy.array( [peakdsq, fit] ))
  return p, fitdata

for i,l in enumerate(ls):
  #rate = evaloss_rate( ps[i], xs[i])
  rate = evaloss_rate( ps[i], l[:,1]/1e3)
  sqrtrate = numpy.power( numpy.abs(rate[:,1]), 0.5)
  #ax2.plot( rate[:,0], sqrtrate, '-', color=c[i])
  peakdsq = numpy.power(l[:,3]/1e12,2.)
  ax2.plot( peakdsq , sqrtrate, '.', color=c[i], ms=18, mew=0.8, mec='black')
  #A = numpy.vstack( [ peakdsq, numpy.ones(len(peakdsq))]).T
  #losscoeff, yint = numpy.linalg.lstsq(A, sqrtrate)[0]
  #ax2.plot( peakdsq, losscoeff*peakdsq + yint, '-', color=c[i])
  losscoeff, losscoeff_fit = fitcoeff( peakdsq, sqrtrate, [1.3] )
  print "Loss coefficient %d: %e" % (i, losscoeff)
  if i == 4:
    #losscoeff = losscoeff - 2.8e-2
    ax2.plot( peakdsq, (losscoeff)*peakdsq , '-', color=c[i])
  
  #ax2.plot( l[:,1], l[:,2], '.', color=c[i])
  ax1.plot( es[i][:,0], es[i][:,1], '-',color=c[i])
  #ax1.plot( xs[i], fs[i](xs[i]) , '-', color=c[i])
  ax1.plot( l[:,1]/1e3, l[:,2]/1e6, '.', color = c[i], markersize=18, mew=0.8, mec='black') 
  #ax1.plot( x, fs[i](x), '-', color = c[i])

#ax1.legend( [p974,p627], [r'\figureversion{lf,tab}$-389$ MHz from state $|2\rangle$',\
#                          r'\figureversion{lf,tab}$-42$ MHz from state $|2\rangle$'], \
#            loc='lower right', bbox_to_anchor=(0.97,0.01) , numpoints = 1, handlelength=0.6, handletextpad=0.5)

ax1.yaxis.set_major_formatter( matplotlib.ticker.FormatStrFormatter(r'%.2f'))
ax1.xaxis.set_major_formatter( matplotlib.ticker.FormatStrFormatter(r'%d'))

for tick in ax1.xaxis.get_major_ticks():
  tick.label.set_fontsize(20)
for tick in ax1.yaxis.get_major_ticks():
  tick.label.set_fontsize(20)

for tick in ax2.xaxis.get_major_ticks():
  tick.label.set_fontsize(20)
for tick in ax2.yaxis.get_major_ticks():
  tick.label.set_fontsize(20)

#ax1.axvline(x=0, linewidth=1.5, color='red')
#ax1.axvline(x=78,linewidth=1.5, color='blue')

ax2.set_ylim(0,2.5)

ax1.spines["bottom"].set_linewidth(2)
ax1.spines["top"].set_linewidth(2)
ax1.spines["left"].set_linewidth(2)
ax1.spines["right"].set_linewidth(2)

ax2.spines["bottom"].set_linewidth(2)
ax2.spines["top"].set_linewidth(2)
ax2.spines["left"].set_linewidth(2)
ax2.spines["right"].set_linewidth(2)

#ax1.set_title(r'Bragg pulse duration = 1 ms,  Bragg beam power = 0.33 mW', fontsize=16)
ax1.set_xlabel(r"Time (s)", fontsize=20, labelpad=16)
ax1.set_ylabel(r"Number($10^{6}$)", fontsize=20, labelpad=20)

ax2.set_xlabel(r"Peak density squared ($10^{24}$ cm$^{-6}$)", fontsize=20, labelpad=16)
ax2.set_ylabel(r"Loss rate   $\ \ \dot{N}/N\ (ms^{-1})$ ", fontsize=20, labelpad=20)

fig.savefig( 'odt_lifetime.png' , dpi=140)
fig2.savefig( 'odt_losscoeff.png' , dpi=140)



