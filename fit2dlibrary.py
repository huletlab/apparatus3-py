from scipy import optimize
import numpy
import matplotlib.pyplot as plt
import inspect
import pprint

class fits:
   def __init__(self, function):
     self.function = function

#-------------------------------------------------------------------------------#
#
#  DIFFERENT TYPES OF FITS ARE DEFINED HERE
#
#-------------------------------------------------------------------------------#
# Currently accepts fits of maximum 5 parameters

fitdict = {}

#---------------------- 2D GAUSSIAN WITH ROTATION
# p0 = amplitude
# p1 = center 0
# p2 = center 1
# p3 = 1/e^2 radius 0
# p4 = 1/e^2 radius 1
# p5 = offset
# p6 = phi
gaus2dphi = fits( lambda p,x,y : p[0]*numpy.exp( -2.0* ( ( -(x-p[1])*numpy.sin(p[6]) + (y-p[2])*numpy.cos(p[6])  )  /p[4] )**2  \
                                              -2.0* ( ( (x-p[1])*numpy.cos(p[6]) + (y-p[2])*numpy.sin(p[6])  )  /p[3] )**2  ) \
                              + p[5] )
gaus2dphi.fitexpr = '0:ampl, 1:c0, 2:c1, 3:w0, 4:w1, 5:offset, 6:phi'
fitdict['Gaussian2Dphi'] = gaus2dphi

#---------------------- 2D GAUSSIAN
# p0 = amplitude
# p1 = center 0
# p2 = center 1
# p3 = 1/e^2 radius 0
# p4 = 1/e^2 radius 1
# p5 = offset
gaus2d = fits( lambda p,x,y : p[0]*numpy.exp( -2.0* (  (x-p[1])  /p[3] )**2  \
                                              -2.0* (  (y-p[2])  /p[4] )**2  ) \
                              + p[5] )
gaus2d.fitexpr = '0:ampl, 1:c0, 2:c1, 3:w0, 4:w1, 5:offset'
fitdict['Gaussian2D'] = gaus2d

#-------------------------------------------------------------------------------#
#
#  THE FITTING PROCEDURES ARE DEFINED BELOW
#
#-------------------------------------------------------------------------------#
def mask_function( p, mask, function):

        mask = numpy.array(mask)
        p = numpy.array(p)
        mask_matrix = []
        counter = 0

        for i ,m in enumerate(mask):
                if m == 0:
                        mask_matrix.append([ 0 for j in range(mask.sum())])
                elif m ==1 :
                        mask_matrix.append([ 1 if counter==j else 0 for j in range(mask.sum())])
                        counter = counter +1

        mask_matrix = numpy.transpose(numpy.matrix(mask_matrix))
        function_masked = lambda x_m,p_m : function (x_m,(numpy.array(p_m)*mask_matrix+(1-mask)*p).tolist()[0])

        return function_masked,mask_matrix

def fit_mask_function(p,data,mask,function):

	mask = numpy.array(mask)
	p = numpy.array(p)
	mask_fun, mask_matrix = mask_function(p,mask,function)
	pfit,fiterror = fit_function((numpy.array(p)*mask_matrix.transpose()).tolist()[0],data,mask_fun)
	print numpy.array(p)*mask_matrix.transpose(),numpy.array(pfit).reshape(1,len(pfit)),mask_matrix,mask,p
	pfit_unmask = numpy.array(pfit.reshape(1,len(pfit))*mask_matrix + (1-mask)*p)
	fiterror_unmask = numpy.array(fiterror.reshape(1,len(pfit))*mask_matrix)

	return pfit_unmask.reshape(5,1), fiterror_unmask.reshape(5,1)

def fit_function(p,data,function):
    # Chekck the length of p
    #pLen = len(inspect.getargspec(function)[0])-1
    #p0 = p[0:pLen]
    p0 = p
   

    errfunc = lambda p, dat: numpy.ravel(function( p, *numpy.indices(dat.shape)  ) - dat)
    pfit, pcov, infodict, errmsg, success = optimize.leastsq( errfunc, p0, args=(data), full_output=1)
    #pfit, pvariance = optimize.curve_fit(function,datax,datay,[p0])

   
    # Estimate the confidence interval of the fitted parameter using
    # the bootstrap Monte-Carlo method
    # http://phe.rockefeller.edu/LogletLab/whitepaper/node17.html
    residuals = errfunc( pfit, data )
    s_res = numpy.std(residuals)
    ps = []
    for i in range(200):
      randomdata = data+numpy.random.normal(0., s_res, data.shape)
      randomfit, randomcov = optimize.leastsq( errfunc, pfit, args=(randomdata), full_output=0)
      ps.append( randomfit ) 
    ps = numpy.array(ps)
    mean_pfit = numpy.mean(ps,0)
    # 2sigma confidence interval is = 95.44 %
    # 1sigma is only 68.3 %
    sigmas = 1.0
    err_pfit = sigmas * numpy.std(ps,0) 
    
    # Below is the old estimation of the fit parameter errors
    # This uses the covariance, the code was copied from the
    # scipy implementation of optimize.curve_fit
    # https://github.com/scipy/scipy/blob/master/scipy/optimize/minpack.py#L247
    if (len(numpy.ravel(data)) > len(p0)) and pcov is not None:
        s_sq = errfunc(pfit,data).sum()/(len(numpy.ravel(data))-len(p0))
        pcov = pcov * s_sq
    else:
        pcov = numpy.inf
    error=[]
    for i in range(len(pfit)):
        try:
          error.append( numpy.absolute(pcov[i][i])**0.5)
        except:
          #print "A proper fit error could not be obtained!"
          #print "pfit = ", pfit
          #print "pcov = ", pcov
          error.append( 0.00 )
    
    # By default the bootstrap estimation is used for confidence intervals
    pfit = mean_pfit.tolist()
    error = err_pfit.tolist()

    # Was trying to return same length of pfit, take out by Ernie 08/20/12 
    pfit = numpy.array(pfit)  #numpy.append(numpy.array(pfit),numpy.zeros(5-len(p0))).reshape(5,1)
    error = numpy.array(error)#numpy.append(numpy.array(error),numpy.zeros(5-len(p0))).reshape(5,1)
    
    return pfit,error

def plot_function(p,datax,function, xlim = None):
    p0 = p
    
    if xlim == None: 
      x = numpy.linspace(numpy.min(datax), numpy.max(datax), 200)
    else: 
      x = numpy.linspace(xlim[0], xlim[1], 200)
    y = function(x,p0)
    return x, y


def fake_data(p,datax,function):
    y = function(datax,p)
    return datax, y
    

def test_function(p,function):
	# generate random data
	ax=numpy.linspace(0,3,12)
	# print p
	x,dat = fake_data( p, ax, function)
	ay = numpy.array(dat)
	noise = 200*(numpy.random.rand(ax.shape[0])-0.5)
	noisydat = ay+noise
        randomdata = numpy.transpose(numpy.array((ax,noisydat)))

	# fit noisy data, starting from a random p0
        p0 = p + p*(0.2*(numpy.random.rand(len(p))-0.5))
        print '          Fake data = ' + str(p)
        print 'Starting parameters = ' + str(p0)
	pFit , error = fit_function( p0, randomdata,function)
        print '         Fit result = ' + str(pFit)

	# Get a plot of the fit results
	fitX, fitY = plot_function(pFit, randomdata[:,0],function)
	# Show the plot on screen

	plt.plot(ax, noisydat,'.')
	plt.plot(fitX,fitY,'-')
	plt.show()

from traits.api import *
from traitsui.api import View, Item, Group, HGroup, VGroup, HSplit, VSplit,Handler, CheckListEditor, EnumEditor, ListStrEditor,ArrayEditor, spring, BooleanEditor,ListEditor

import pickle

class Fits(HasTraits):
    """ Object used to do fits to the data
    """
    doplot = Bool(False, desc="plot?: Check box to plot with the current params", label="plot?")
    dofit = Bool(False, desc="do fit?: Check box to enable this fit", label="fit?")
    fitexpr = Str(label='f(x)=')
    func = Enum(fitdict.keys())
    x0 = Float(-1e15, label="x0", desc="x0 for fit range")
    xf = Float(1e15, label="xf", desc="xf for fit range")
    
    y0 = Float(-1e15, label="y0", desc="y0 for fit range")
    yf = Float(1e15, label="yf", desc="yf for fit range")
    fit_mask = List(Bool(True,editor=BooleanEditor(mapping={"yes":True, "no":False})),[True,True,True,True,True])   

    a0 = Array(numpy.float,(5,1),editor=ArrayEditor(width=-100))
    a = Array(numpy.float,(5,1),editor=ArrayEditor(width=-100))
    ae = Array(numpy.float,(5,1),editor=ArrayEditor(width=-100))
	
    traits_view = View(
                    Group(Group(
                       Item('doplot'),
                       Item('dofit'),
                       Item('func'),
                        orientation='horizontal', layout='normal'), 
                        Group(
                       Item('x0'),
                       Item('xf'), 
                       orientation='horizontal', layout='normal'),
                                               Group(
                       Item('y0'),
                       Item('yf'), 
                       orientation='horizontal', layout='normal'), 
                    Group(
                       Item('fitexpr',style='readonly')),
                    Group(
                       Item('a0'),
                       Item('a'),
					   Item('ae'),
			Item('fit_mask', style='custom',editor = ListEditor()),
                       orientation='horizontal'),),
                       dock='vertical',
               )
               
    def limits(self, data):
        lim=[]
        for p in data:
            
            if p[0] < self.xf and p[0] > self.x0 and p[1] > self.y0 and p[1] < self.yf:
                lim.append([p[0],p[1]])
        return numpy.asarray(lim), len(lim)
        
            
    def _setfitexprs_(self):
        try: 
          self.fitexpr = fitdict[self.func].fitexpr
        except:
          print "No fit called %s exists! Program will exit." % self.func
          exit(1)
                              
    def fit(self,data):
	mask =  [ 1 if i else 0 for i in self.fit_mask]
        fitdata, n = self.limits(data)
        if n == 0:
            print "No points in the specified range [x0:xf], [y0:yf]"
            return None,None
        f = fitdict[self.func]
        if not self.dofit:
          print "Evaluating %s" % self.func
          return plot_function(self.a0[:,0] , fitdata[:,0], f.function)
        else:
          print "Fitting %s" % self.func
          self.a, self.ae=fit_mask_function(self.a0[:,0],fitdata,mask,f.function)
          return plot_function(self.a[:,0] , fitdata[:,0],f.function)
           
if __name__ == "__main__":
        print ""
	print "------ Functions in Fit Library ------"
        for key in fitdict.keys():
          print key
      
       
	print ""
	print "------ Testing fitlibrary.py ------"
	print ""

	test_function([1000,700],fitdict['Temperature'].function)



    
