#!/usr/bin/python
import sys
import numpy as np

import matplotlib 
matplotlib.use('Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

sys.path.append('/lab/software/apparatus3/py')
import qrange, statdat, fitlibrary
from uncertainties import ufloat,unumpy

from scipy import stats
import coldens
from colorChooser import rgb_to_hex, cmapCycle

import os
import argparse


magnif = 1.497 # um per pixel
lattice_d = 0.532

#magnif = 1.497 / 1.03228 # um per pixel
#lattice_d = 0.532


from configobj import ConfigObj

import numpy
from scipy import optimize
import math

def fitGaussian(xdata,ydata):
	xdata = numpy.array(xdata)
	ydata = numpy.array(ydata)
	erf = numpy.vectorize(math.erf)
	p0 = [ydata.min()-ydata.max(),np.argmin(ydata),2,ydata[0]]
	fitfunc = lambda p, x: p[0]*np.exp(-2.0*((x-p[1])/p[2])**2)+p[3]
	
	errfunc = lambda p, x, y: fitfunc(p,x) - y
	
	p, cov, infodict, errmsg, success = optimize.leastsq(errfunc, p0[:], args=(xdata, ydata), full_output=1)
	fitX = [(xdata[-1]-xdata[0])/500.0*i +xdata[0] for i in range(500)]
	fitY = fitfunc(p,fitX)
	
	return p, fitX,fitY



def fitDoubleGaussian(xdata,ydata):
	xdata = numpy.array(xdata)
	ydata = numpy.array(ydata)
	erf = numpy.vectorize(math.erf)
	p0 = [ydata.max()-ydata.min(),np.argmax(ydata),30,(ydata.min()-ydata.max())*0.05,np.argmax(ydata),1.0,ydata[0]]
	fitfunc = lambda p, x: p[0]*np.exp(-2.0*((x-p[1])/p[2])**2)+p[3]*np.exp(-2.0*((x-p[1])/p[5])**2)+p[6]
	
	errfunc = lambda p, x, y: fitfunc(p,x) - y

	cons = ({'type': 'ineq',
         	'fun' : lambda x: np.array([-1*x[5]]),
         	'jac' : lambda x: np.array([0.0,0,0,0,0 ,-1.0,0])})
	
	p, cov, infodict, errmsg, success = optimize.leastsq(errfunc, p0[:], args=(xdata, ydata), full_output=1)
	p[4]=p[1]
	fitfunc = lambda p, x: p[0]*np.exp(-2.0*((x-p[1])/p[2])**2)+p[3]*np.exp(-2.0*((x-p[4])/p[5])**2)+p[6]
	errfunc = lambda p, x, y: fitfunc(p,x) - y
	
	p, cov, infodict, errmsg, success = optimize.leastsq(errfunc, p[:], args=(xdata, ydata), full_output=1)
	
	#p, cov, infodict, errmsg, success = optimize.fmin_slsqp(errfunc, p0[:], args=(xdata, ydata), full_output=1,constraints=cons)
	fitX = [(xdata[-1]-xdata[0])/500.0*i +xdata[0] for i in range(500)]
	fitY = fitfunc(p,fitX)
	
	return p, fitX,fitY


def cut_single( datadir, shot,rows,cols,doubleG=False,**kwargs):

    # Find out the aspect ratio 
    inifile = datadir + 'report' + "%04d"%shot + '.INI'
    report = ConfigObj(inifile) 



    figTest = Figure( figsize=(6,4 ) )
    canvas = FigureCanvas( figTest ) 
    gsTest = matplotlib.gridspec.GridSpec( 1,2, 
                wspace=0.5, hspace=0.5,\
                left=0.1, right=0.90, bottom=0.20, top=0.80 ) 
    
    cddata = np.loadtxt( os.path.join( datadir, '%04d_column.ascii'%shot ) ) 
    print len(cddata)
    cddata_cut = cddata [rows[0]:rows[1],cols[0]:cols[-1]] 	 
    
    coldens.PlotCD(figTest, gsTest[0,0], \
        cddata = cddata_cut,\
        dirpath=datadir,\
        title=None)

    oneD =  cddata_cut.sum(axis=0)
    print "Sum the whole cut" , oneD.sum()	
    oneDX= range(len(oneD))
    ax1D = figTest.add_subplot(gsTest[0,1])
    if not doubleG:
    	p,fitX,fitY= fitGaussian(oneDX,oneD)
    else:    	
	p,fitX,fitY= fitDoubleGaussian(oneDX,oneD)
    ax1D.plot(oneDX,oneD)	
    ax1D.plot(fitX,fitY)	
    ax1D.set_xlabel("Pixel")
    ax1D.set_ylabel("Summed Column Density")	
    ax1D.grid(True)	
    if not doubleG:
    	ax1D.set_title( "Waist is %.2e um."%(p[2]*magnif),y=1.08)
    	print "Waist is %.2e um."%(p[2]*magnif)
    	print p
    else:
    	ax1D.set_title( "Waists are\n %.2e,%.2e um."%(p[2]*magnif,p[5]*magnif),y=1.1)
	print "Waists are\n %.2e,%.2e um."%(p[2]*magnif,p[5]*magnif)
	print "Amp=%d,%d"%(p[0],p[3])
	
    #xscale = magnif*np.sqrt(AR)
    #yscale = (magnif/lattice_d)**-3 


    #savedir = 'abelsingle/{:03d}/{:.2f}/'.format( int(aS), gr ) 
    savedir = 'plots/'
    if not os.path.exists( savedir ) :
        os.makedirs( savedir )  

    base = '1D__shot{:04d}'.\
             format( int(shot) ) 
  
    fname = base + '.png' 
    canvas.print_figure( savedir + fname , dpi=250)
    # plt is evil, leaks memory.  switched to Figure and canvas approach
    #plt.close(figTest)
  
    

if __name__ == '__main__':
#    datadir = '/lab/data/app3/2015/1506/150617/'
#    shot = 5282

	parser = argparse.ArgumentParser()
	parser.add_argument("shot",help="shot to do 1D")
	parser.add_argument('--dG', dest='dG', action='store_true')
	parser.set_defaults(dG=False)
	args = parser.parse_args()
	shot = int(args.shot)
	dG = bool(args.dG)
	datadir = "./"
	rows=[0,-1]
	cols=[0,-1]
	cut_single( datadir, shot,rows,cols, doubleG = dG, printline=True ) 


#
