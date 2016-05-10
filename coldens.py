
import os
import numpy as np

from matplotlib import cm
from matplotlib import pyplot as plt

def average_cd( datadir, shots):
    shape = None
    cds = []
    for sh in shots: 
        cd = np.loadtxt( os.path.join( datadir, '%04d_column.ascii'%sh ) ) 
        if shape is None: 
            shape = cd.shape 
            cds.append(cd)
        elif cd.shape != shape:
            print "Error, found a column density matrix with"
            print "mismatching dimensions in"
            print datadir, " shot #", sh
            print "   shape = ", shape
            print "cd.shape = ", cd.shape
        else: 
            cds.append(cd)
    #print "Collected coldens for %d shots" % len(cds)

    ave = np.zeros_like( cds[0] )
    num = 0  
    for cd in cds: 
        ave = ave + cd 
        num += 1
    if num > 0: 
        ave = ave / num
    else:
        print "Error when doing average. Divide by zero."
        print "Check that there are any valid shots. "
    return ave

import scipy.ndimage as ndimage 
def cdpeak( cddata, **kwargs):
    cdfiltered = ndimage.gaussian_filter( cddata, sigma=3, order=0) 
    assert cdfiltered.shape == cddata.shape
    peakindex = np.unravel_index( cdfiltered.argmax(), cdfiltered.shape ) 
    #print "Peak at ", peakindex, " in ", cddata.shape
    return peakindex 
     
     
def cdplot( ax, cddata, **kwargs):
    imkwargs = {}
    if 'vmin' in kwargs.keys():
        imkwargs['vmin'] = kwargs['vmin']
    if 'vmax' in kwargs.keys():
        imkwargs['vmax'] = kwargs['vmax']
    
   
    im = ax.imshow(cddata, cmap=cm.spectral, aspect='equal', **imkwargs)

    if not kwargs.get('scale',False):
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])


    # Here I find the size of the axes in inches. I use it to 
    # scale up the X and Y cut axes . 
    Yfig = ax.get_figure() 
    Ysz = Yfig.get_size_inches()  
    points0 = ax.get_position().get_points()
    H0 = (points0[1,0] - points0[0,0] ) 
    V0 = (points0[1,1] - points0[0,1] ) 
    H0in = H0 * Ysz[0]
    V0in = V0 * Ysz[1]

    # Here is the aspect ratio of the data 
    AR = float(cddata.shape[0]) / float( cddata.shape[1] )

    axesverbose = False
   
    if axesverbose:
        print "--CONSTRUCTING AXES FOR DENSITY INSPECT--"
        print "ax size in inches : ( HOR=%.2f, VER=%.2f )" % (H0in, V0in)
        print "data aspect ratio : %.2f " % AR

        from matplotlib.patches import Rectangle

        ax.add_patch( Rectangle( (0.,0.), \
            1.,1., clip_on=False,\
            transform = ax.transAxes, fc='red', alpha=0.8 ) )

        from matplotlib.patches import Circle
        circle = Circle((0.,0.), 0.05, fc='black', transform=ax.transAxes,\
                        clip_on=False) 
        ax.add_patch(circle)
        circle = Circle((0.,1.), 0.05, fc='black', transform=ax.transAxes,\
                        clip_on=False) 
        ax.add_patch(circle)
        circle = Circle((1.,0.), 0.05, fc='black', transform=ax.transAxes,\
                        clip_on=False) 
        ax.add_patch(circle)
        circle = Circle((1.,1.), 0.05, fc='black', transform=ax.transAxes,\
                        clip_on=False) 
        ax.add_patch(circle)

    # Here I get the corners of the ax in figure coords
    x0,y0 = ax.figure.transFigure.inverted().transform( \
            ax.transAxes.transform((0.,0.)))
    x1,y1 = ax.figure.transFigure.inverted().transform( \
            ax.transAxes.transform((1.,1.)))
    dx = x1-x0; dy = y1-y0
    border = min(dx/40., dy/40.)
   
    #  Here I get the centers of the image ax
    xc = x0 + dx/2. 
    yc = y0 + dy/2. 

    if axesverbose:
        print "x0, y0 = %.2f, %.2f" % (x0,y0)
        print "x1, y1 = %.2f, %.2f" % (x1,y1)
        print "dx = %.2f " % dx 
        print "dy = %.2f " % dy
        print "xc = %.2f " % xc
        print "yc = %.2f " % yc 

    # Image axes gets clipped on the sides 
    if V0in/H0in  < AR :
        sc = (V0in / AR) / H0in
        if axesverbose:
            print "Image axes gets clipped on the sides."
            print "Using xsc = %.2f to make cut axes." % sc
        
        axXcut = ax.figure.add_axes( [\
            xc - dx/2.*sc,\
            y1 + border,\
            dx*sc ,\
            dy/3.] ) 
      
        axYcut = ax.figure.add_axes( [\
            xc - dx/2.*sc - dx/3. - border,\
            y0,\
            dx/3. ,\
            dy] )
    
        cbarax = ax.figure.add_axes( [\
            xc + dx/2.*sc + border ,\
            y0,\
            dx/10. ,\
            dy ] )

    # Image axes gets clipped on top and bottom
    else:
        sc = (H0in * AR) / V0in
        if axesverbose:
            print "Image axes gets clipped on top and bottom."
            print "Using ysc = %.2f to make cut axes." % sc
        
        axXcut = ax.figure.add_axes( [\
            x0,\
            yc + dy/2.*sc + border,\
            dx ,\
            dy/3.] ) 
      
        axYcut = ax.figure.add_axes( [\
            x0 - dx/3. - border,\
            yc - dy/2.*sc,\
            dx/3. ,\
            dy*sc] )
    
        cbarax = ax.figure.add_axes( [\
            x1 + border ,\
            yc - dy/2.*sc,\
            dx/10. ,\
            dy*sc ] )
    
    
    axYcut.set_ylim( ax.get_ylim())
    axXcut.set_xlim( ax.get_xlim())


    cbar = plt.colorbar(im, cax=cbarax)
    cbar.ax.tick_params(labelsize=7)
    return im, axXcut, axYcut




def average_azcd( datadir, shots, **kwargs):
    Azs = []  
    xsets = []
    for sh in shots:
        fpath = os.path.join( datadir, '%04d_datAllAzimuth.AZASCII'%sh )
        Az = np.loadtxt( fpath ) 
        Azs.append(Az)
        xsets.append( set(Az[:,0].tolist()) ) 
    
    # First get all the r's that the various Az's have in common
    rvals = np.array(sorted(list(set.intersection( *xsets))))
    # Restrict the data to half of the rvals
    rvals = rvals[ : len(rvals)/2]

    # Then go ahead and sum up the average for the coldens value
    yvals = np.zeros_like(rvals)
    num = 0
    for Az in Azs:
         yvals = yvals + Az[:,1][ np.where( np.in1d( Az[:,0], rvals) ) ]  
         num +=1 
    if num > 0: 
        yvals = yvals / num
    else:
        print "Error when doing average. Divide by zero."
        print "Check that there are any valid shots. "

    # In case one wants to return the symmetrized azim average
    #rvals = np.hstack(( (-1.*rvals)[::-1], rvals ))
    #yvals = np.hstack(( (yvals)[::-1], yvals ))

    trimStart = kwargs.get('trimStart', None)
    if trimStart is not None:
        rvals = rvals[ trimStart:] 
        yvals = yvals[ trimStart:] 

    return (rvals,yvals)



def cutplotROWS( ax, cddata, ri=38,rf=42, parent=None):
    #print "Making cut plot X"
    #print "\tshape of cddata = ",cddata.shape
    cutdat = cddata[ range(ri,rf),:].sum(axis=0)
    num = len( range(ri,rf) ) 
    cutdat = cutdat /num 
    ax.plot(  cutdat ) 
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.grid(alpha=0.3)
    parent.axhline( y=(ri+rf)/2., color='black', lw=0.5, alpha=0.5)
    return (cutdat.min(), cutdat.max()) 

def cutplotCOLS( ax, cddata, ci=38,cf=42, parent=None):
    cutdat = cddata[ :,range(ci,cf)].sum(axis=1) 
    num = len( range(ci,cf) ) 
    cutdat = cutdat /num 
    index  = range(len(cutdat))
    ax.plot(  cutdat, index )
    ax.invert_xaxis() 
    ax.invert_yaxis() 
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.grid(alpha=0.3)
    parent.axvline( x=(ci+cf)/2., color='black', lw=0.5, alpha=0.5)
    return (cutdat.min(), cutdat.max()) 

def match_ylim( ylims, axRow, axCol):
    ylims = np.array(ylims) 
    ymin = ylims[:,0].min()
    ymax = ylims[:,1].max()
    axRow.set_ylim( ymin, ymax)
    axCol.set_xlim( ymax, ymin)
        
    

import matplotlib
 
def makeCutAxes( fig, gs): 
    gsSUB = matplotlib.gridspec.GridSpecFromSubplotSpec( 2,2,\
                subplot_spec = gs,  wspace = 0.01, hspace=0.01,\
                width_ratios=[1,3], height_ratios=[1,3])
    ax     = fig.add_subplot( gsSUB[1,1])
    axXcut = fig.add_subplot( gsSUB[0,1] )
    axYcut = fig.add_subplot( gsSUB[1,0])

    # Here I try to find the sizes of the figure and outer grid
    # to make the y-axis match the image plot 
    Yfig = axYcut.get_figure() 
    Ysz = Yfig.get_size_inches()  
    points0 = ax.get_position().get_points()
    H0 = (points0[1,0] - points0[0,0] ) 
    V0 = (points0[1,1] - points0[0,1] ) 
    H0in = H0 * Ysz[0]
    V0in = V0 * Ysz[1]

    # To accomodate the aspect ratio of the image
    # matplotlib will scale down the y-axis by  
    alpha = H0in / V0in
    points1 = axYcut.get_position().get_points()
    H1 = (points1[1,0] - points1[0,0] ) 
    V1 = (points1[1,1] - points1[0,1] )
    newbox = matplotlib.transforms.Bbox.from_bounds( \
                 points1[0,0], points1[0,1] + (V1-V1*alpha)/2,\
                 H1, V1*alpha )
    axYcut.set_position(newbox) 
    return ax, axXcut, axYcut 

def makeCDAxes( fig, gs, **kwargs): 
    gsSUB = matplotlib.gridspec.GridSpecFromSubplotSpec( 2,3,\
                subplot_spec = gs,  wspace = 0.01, hspace=0.01,\
                width_ratios=[1,3,0.4], height_ratios=[1,3])
    ax     = fig.add_subplot( gsSUB[1,1])
    gsRect =  gs.get_position(fig).bounds

    # Uncomment to draw rectangle around gs
    #from matplotlib.patches import Rectangle
    #ax.add_patch( Rectangle( (gsRect[0],gsRect[1]), \
    #    gsRect[2], gsRect[3], clip_on=False,\
    #    transform = fig.transFigure, fc='gray', alpha=0.2 ) )

    if kwargs.get('title',None) is not None:
        fig.text( gsRect[0]+gsRect[2]/2. , gsRect[1]+gsRect[3]*1.05, \
              kwargs.get('title',None), ha='center', fontsize=8 ) 
    
    return ax


def sumEllipse( cddata, edict, **kwargs):
    C, R = np.meshgrid( range(cddata.shape[1]), range(cddata.shape[0]) )
 
    c0 = edict['xy'][0]
    r0 = edict['xy'][1]
  
    # First rotate the coordinates:
    th = edict['angle']*np.pi / 180. 
    Crot =  (C-c0)*np.cos(th) + (R-r0)*np.sin(th)
    Rrot = -(C-c0)*np.sin(th) + (R-r0)*np.cos(th)

    Cw = edict['width']/2.
    Rw = edict['height']/2. 
    
    mask =  np.power( Crot/Cw, 2.) + np.power( Rrot/Rw, 2.)  < 1.
    mask = mask.astype(int) 
   
    #np.set_printoptions(threshold=np.nan)
    #print mask
    region = mask * cddata

    ax = kwargs.pop( 'plotax', None)
    if ax is not None:
        im = ax.imshow(region, cmap=cm.spectral, aspect='equal',**kwargs)
        #im = ax.imshow(region, cmap=cm.winter, aspect='equal',**kwargs)
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        
        axRect = ax.get_position().bounds
        cbarax = ax.figure.add_axes( [ axRect[0]+axRect[2]+axRect[2]/40., axRect[1], axRect[2]/10., axRect[3] ] ) 
        cbar = plt.colorbar(im, cax=cbarax)
        cbar.ax.tick_params(labelsize=5)

    return np.sum( region )
    

def makeAzAxes( fig, gs): 
    gsSUB = matplotlib.gridspec.GridSpecFromSubplotSpec( 2,1,\
                subplot_spec = gs,  wspace = 0.01, hspace=0.01,\
                width_ratios=[1,1], height_ratios=[1,1])
    ax0    = fig.add_subplot( gsSUB[0,0])
    ax1    = fig.add_subplot( gsSUB[1,0])
    return ax0, ax1


from colorChooser import rgb_to_hex, cmapCycle
    
def azplot( ax, azdata, **kwargs):
    l = kwargs.get('labelstr', None)
    c = kwargs.get('col','black') 
    xscale = kwargs.get('xs',1.0) 
    yscale = kwargs.get('ys',1.0) 

    xsdat = xscale * azdata[0] 
    ysdat = yscale * azdata[1] 

    ax.plot( xsdat ,ysdat, '-', marker='.',\
            ms=5, color=c, label=l )
    return xsdat, ysdat 
  

from inverabel import inverAbelDat

def azabel( ax, axHist, azdata, **kwargs):
    l = kwargs.get('labelstr', None)
    c = kwargs.get('col',None)
    which = kwargs.get('which', 'Int' )
 
    cInt = kwargs.get('colInt','blue')
    cRad = kwargs.get('colRad','red')
  
    if c is not None:
        cInt = c; cRad = c;
 
    xscale = kwargs.get('xs',1.0) 
    yscale = kwargs.get('ys',1.0) 
  
    abdat = inverAbelDat( azdata[0], azdata[1] )
   
    tS = kwargs.get('trimStart', 0 )  
    # Direct integral
    datInt = np.column_stack(( xscale*abdat[0][0][tS:],\
                               yscale*abdat[0][1][tS:] ))
    # Radon transform
    datRad = np.column_stack(( xscale*abdat[1][0][tS:],\
                               yscale*abdat[1][1][tS:] ))

    if ax is not None:
        if 'both'==which or 'Int'==which: 
            ax.plot( datInt[:,0], datInt[:,1],'-', marker='.',\
                    ms=5, color=cInt, label=l )
                    #ms=5, color=cInt, label=l+'Integral' )
        if 'both'==which or 'Rad'==which: 
            ax.plot( datRad[:,0], datRad[:,1],':', marker='.',\
                    ms=5, color=cRad, label=l+'Radon' )

    #axHist.hist( datInt[:,1] , bins=10, weights=4*np.pi*np.power(datInt[:,0],2) )

    return datInt, datRad 
            
from scipy import interpolate
from scipy.interpolate import UnivariateSpline

def numDiff( x, y ): 
    #print np.column_stack(( np.diff(x), np.diff(y) )) 
    dy = np.diff(y) / np.diff(x) 
    xd = ( x[1:] + x[:-1] ) / 2. 
    return xd, dy

def abelKDiff( ax, abdat, **kwargs): 
    l = kwargs.get('labelstr', None)
    c = kwargs.get('col','black') 
    # Obtain the derivative of the data directly
    xd, dy = numDiff( abdat[:,0], abdat[:,1] )
    dy = -1.*dy
    x = abdat[:,0][1:]
    y = abdat[:,1][1:]
    trimStart = kwargs.get('trimStart', None)
    if trimStart is not None:
        x  = x[ trimStart:] 
        y  = y[ trimStart:] 
        xd = xd[ trimStart:] 
        dy = dy[ trimStart:] 
    ax.plot( xd, dy, marker='.',\
            ms=5, color=c, label=l )

    axKvsn = kwargs.get('axKvsn', None)
    ydens  = kwargs.get('ydens', None)
    if axKvsn is not None:
        if ydens is None:
            axKvsn.plot( y, dy/xd, marker='.',ms=5, color=c, label=l)
        else:
            xncol = ydens[:,0]
            yncol = ydens[:,1] 
            common = set(xncol.tolist()) & set(x.tolist()) 
            in1=[]; in2=[]; 
            for cval in common:
                in1.append( np.where( xncol == cval )[0][0] )  
                in2.append( np.where( x == cval )[0][0] ) 

            #Sort them by density 
            insort = np.argsort( yncol[in1] ) 

            axKvsn.plot( yncol[in1][insort], dy[in2][insort]/xd[in2][insort],\
                         marker='.',ms=5, color=c, label=l)

def abelKSpline( ax, axAb, axK, abdat, **kwargs): 
    l = kwargs.get('labelstr', None)
    c = kwargs.get('col','black')
 
    xs = np.linspace( abdat[:,0].min(), abdat[:,0].max(), 100 ) 
    # Fit the abel transform data with a spline
    ##spline = UnivariateSpline( abdat[:,0], abdat[:,1] )
    ##ys = spline(xs) 
    tck = interpolate.splrep( abdat[:,0], abdat[:,1] , s=0) 
    ys = interpolate.splev(xs, tck, der=0)
    axAb.plot( xs, ys,\
                   ms=5, color='black', label=l )

    ys1 = -1.*interpolate.splev(xs, tck, der=1 )
 
    tS = kwargs.get('trimStart', None)
    if tS is not None:
        umperpt = abdat[1,0] - abdat[0,0]
        minx = (tS+1) * umperpt
        index = np.where( xs > minx )
        xs = xs[index]
        ys1 = ys1[index] 
        ys = ys [index] 
        
    ax.plot( xs, ys1,\
                   ms=5, color=c, label=l )
        
    axK.plot( xs, ys1/xs,\
                   ms=5, color=c, label=l )
    
    axKvsn = kwargs.get('axKvsn', None)
    if axKvsn is not None:
        axKvsn.plot( ys, ys1/xs, ms=5, color=c, label=l)

from scipy.interpolate import interp1d 
def abelKinterp1d( ax, abdat, **kwargs):
    
    x = abdat[:,0]
    y = abdat[:,1] 
    f = interp1d( x, y, kind = 'cubic' )
    xnew = np.linspace( x.min(), x.max(), 240 ) 
    ynew = f(xnew)
 
    abax = kwargs.get('abax',None) 
    if abax is not None:
        abax.plot( xnew, ynew, '--', color=kwargs.get('col','k') )

    #dndr = -1.*np.diff( ynew ) / np.diff( xnew ) 
    #r  = (xnew[:-1] + xnew[1:] )/ 2.     
    #ax.plot( r, dndr, '-',\
    #         #marker='.', ms=5.,\
    #         color = kwargs.get('col','k'), label = kwargs.get('label',None) )
 
    dndr = -1.*np.diff( y ) / np.diff( x )  * 0.532 # um/site 
    dndr = dndr * kwargs.get('yscale',1.0) 
    r  = (x[:-1] + x[1:] )/ 2.     
    ax.plot( r, dndr, '-',\
             marker='.', ms=5.,\
             color = kwargs.get('col','k'), label = kwargs.get('label',None) ) 

     


# the functio below is obsolete
def abelSpline( axSpline, axSplineDiff, axDiff, abDat, **kwargs):
    l = kwargs.get('labelstr', None)
    c = kwargs.get('col','black') 
    lt = ['-', ':']
    for i,dat in enumerate(abDat):
        # Fit the abel transform data with a spline
        spline = UnivariateSpline( dat[:,0], dat[:,1] )
        #tck = interpolate.splrep( dat[:,0], dat[:,1] , s=0) 
        xs = np.linspace( dat[:,0].min(), dat[:,0].max(), 100 ) 
        #ys = interpolate.splev(xs, tck, der=0)
        ys = spline(xs) 
        axSpline.plot( xs, ys, lt[i],\
                       ms=5, color=c, label=l ) 

        # Obtain the derivative of the spline
        axSplineDiff.plot( xs[1:], np.diff(ys) / np.diff(xs) , lt[i],\
                     ms=5, color=c, label=l ) 

        # Obtain the derivative of the data directly
        xd, dy = numDiff( dat[:,0], dat[:,1] ) 
        axDiff.plot( xd, dy,  lt[i], marker='.',\
                ms=5, color=c, label=l )
        
            

    
########################## 
#### Higher level routines
    
def PlotCD( figure, gs, **kwargs ):
    dirpath = kwargs.pop('dirpath', os.getcwd() ) 
    shots = kwargs.pop('shots', [] ) 

    nrowsAvg = kwargs.pop('nrowsAvg', 1)
    ncolsAvg = kwargs.pop('ncolsAvg', 1)
    cutPos = kwargs.pop('cutPos', 'auto')

    ax = makeCDAxes( figure, gs, **kwargs ) 
    if 'cddata' in kwargs.keys():
        cddata = kwargs.pop('cddata') 
    else:
        cddata =  average_cd( dirpath, shots )
    im, axXcut, axYcut  = cdplot( ax, cddata, **kwargs )
    ax.invert_yaxis()

    if cutPos == 'auto':
        r0, c0 = cdpeak(cddata) 
    else:
        r0, c0 = kwargs.pop('r0', cddata.shape[0]/2), \
                 kwargs.pop('c0', cddata.shape[1]/2)


    cuts_visible = kwargs.pop('cuts_visible', True)
    if not cuts_visible:
        axXcut.set_visible(False)
        axYcut.set_visible(False)
    else:
        cutRowi = r0 -nrowsAvg ; cutRowf = r0 + nrowsAvg
        cutColi = c0 -ncolsAvg ; cutColf = c0 + ncolsAvg
        
        rylim = cutplotROWS( axXcut, cddata, ri=cutRowi, rf=cutRowf, parent=ax)
        cylim = cutplotCOLS( axYcut, cddata, ci=cutColi, cf=cutColf, parent=ax)
        match_ylim( (rylim, cylim), axXcut, axYcut )

    # Set scale on the image and cut axes
    scale = kwargs.pop('scale', False ) 
    if scale is not None:
        if 'magnif' in kwargs.keys():
            magnif = kwargs.get('magnif', 1.0 )
            x0,x1 = ax.get_xlim() 
            y0,y1 = ax.get_ylim() 
            step = 20. 
            ticks = np.linspace( -20*step, 20*step, 41 )
            xtpos = []; xtlab = [] 
            ytpos = []; ytlab = [] 
            for t in ticks:
                if t/magnif > (x0-c0) and t/magnif < (x1-c0):
                    xtpos.append( t/magnif + c0 )
                    xtlab.append( '%d'%int(t) )
                if t/magnif > (y0-r0) and t/magnif < (y1-r0):
                    ytpos.append( t/magnif + r0 )
                    ytlab.append( '%d'%int(t) )
 
            ax.set_xlabel( '$\mu\mathrm{m}$', fontsize=8.)
            ax.set_xticks( xtpos ) 
            ax.set_xticklabels( xtlab , fontsize=7.)
            axXcut.set_xticks( xtpos )
            axXcut.set_xticklabels( [] )   
     
            ax.set_yticks( ytpos ) 
            ax.set_yticklabels( [] )   
            axYcut.set_yticks( ytpos ) 
            axYcut.set_yticklabels( [] )  
    return ax  
             
#            ax.xaxis.set_major_locator(\
#                matplotlib.ticker.MultipleLocator( base = 10./magnif )) 
#            ax.xaxis.set_major_formatter(\
#                matplotlib.ticker.FuncFormatter( lambda x: x*magnif ) ) 


def PlotAZ( figure, **kwargs):
    gs0 = kwargs.pop('gsAz', None ) 
    gs1 = kwargs.pop('gsK', None )
 
    dirpath = kwargs.pop('dirpath', os.getcwd() ) 
    shots = kwargs.pop('shots', [] )

    magnif = kwargs.pop('magnif', 1.497 )
    lattice_d = kwargs.pop('lattice_d', 0.532 )  

    # MAKE AZIMUTHAL, ABEL PLOTS 
    gsAZ = matplotlib.gridspec.GridSpecFromSubplotSpec(2,1,\
               subplot_spec=gs0, wspace=0.01, hspace=0.05) 
    axAz = figure.add_subplot( gsAZ[0,0] ) 
    axAb = figure.add_subplot( gsAZ[1,0] )

    azdata = average_azcd( dirpath, shots, trimStart=0)
    azplot( axAz, azdata,\
            xs=magnif, ys=(magnif/lattice_d)**-2,\
            labelstr='azimuthal', col='blue'  )

    abDat = azabel( axAb, None, azdata, trimStart=1,\
                    xs=magnif, ys=(magnif/lattice_d)**-3,\
                    labelstr='abel', colInt='blue', colRad='red' )


    axAz.set_xticklabels(())
    maxX = max(axAz.get_xlim()[1], axAb.get_xlim()[1] )
    axAz.set_ylabel('$n_{\mathrm{c}}(r)$', rotation=0, fontsize=14, labelpad=5)
    axAb.set_ylabel('$n(r)$', rotation=0, fontsize=14, labelpad=5)

    # MAKE COMPRESSIBILITY PLOTS
    gsK = matplotlib.gridspec.GridSpecFromSubplotSpec(2,1,\
               subplot_spec=gs1, wspace=0.01, hspace=0.05) 
    axnDer = figure.add_subplot( gsK[0,0] ) 
    axK = figure.add_subplot( gsK[1,0] )

    axnDer.set_xticklabels(())
    
    abelKDiff( axnDer, abDat[0], trimStart=2 )  
    abelKSpline( axnDer, axAb, axK, abDat[0], trimStart=2 ) 
 
    # Decorate axes  
    axnDer.set_ylabel(r"$-n'(r)$", rotation=0, fontsize=10, labelpad=-3)
    axK.set_ylabel(r"$-\frac{n'(r)}{r}$", rotation=0, fontsize=14, labelpad=1)
    
    for ax in [axAb, axK]:
        ax.set_xlabel('$\mu\mathrm{m}$', labelpad=4, fontsize=11)
    
    for ax in [axAz, axAb, axnDer, axK]:
        ax.set_xlim(-1., maxX) 
        ax.grid(alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(prune='lower'))
        ax.legend( loc='best', numpoints=1, prop={'size':6},\
                   handlelength=1.1, handletextpad=0.5 )

def PlotAZsimple( figure, **kwargs):
    gs0 = kwargs.pop('gsAz', None ) 
 
    dirpath = kwargs.pop('dirpath', os.getcwd() ) 
    shots = kwargs.pop('shots', [] )

    magnif = kwargs.pop('magnif', 1.497 )
    lattice_d = kwargs.pop('lattice_d', 0.532 ) 

    doabel = kwargs.pop('doabel', False) 

    # SETUP AXES 
    axs = []
    if not doabel: 
        axAz = kwargs.pop( 'axAz', None)
        if axAz is None:
            gsAZ = matplotlib.gridspec.GridSpecFromSubplotSpec(1,1,\
                   subplot_spec=gs0, wspace=0.01, hspace=0.05) 
            axAz = figure.add_subplot( gsAZ[0,0] )
        axs = [axAz] 
    

    else:
        axAz = kwargs.pop( 'axAz', None)
        axAb = kwargs.pop( 'axAb', None)
        if axAz is None:
            gsAZ = matplotlib.gridspec.GridSpecFromSubplotSpec(2,1,\
                   subplot_spec=gs0, wspace=0.01, hspace=0.05) 
            axAz = figure.add_subplot( gsAZ[0,0] )
            axAb = figure.add_subplot( gsAZ[1,0] )
        axs = [ axAz , axAb ] 

    # MAKE AZIMUTHAL PLOT
    azdata = average_azcd( dirpath, shots, trimStart=0)
    
    yscale = (magnif/lattice_d)**-2 
    aznorm = kwargs.pop('aznorm', None)
    if aznorm:
        norm = np.mean( azdata[1][:aznorm['pts'] ] )
        azdata = [azdata[0], azdata[1] * aznorm['norm'] / norm]
        yscale=1.
       
    axAz.plot(magnif*azdata[0],yscale*azdata[1], '-', marker='.',\
            ms=3, color='black', label=kwargs.pop('azlabel',None) )
    axAz.set_ylabel('$n_{\mathrm{c}}(r)$', rotation=0, fontsize=14, labelpad=5)
    aztitle = kwargs.pop('aztitle',None)
    if aztitle:
        axAz.text( 0., 1., aztitle, transform=axAz.transAxes, ha='left',\
                   va='bottom')


    # MAKE ABEL PLOT 
    if doabel:
        abDat = azabel( axAb, None, azdata, trimStart=1,\
                    xs=magnif, ys=(magnif/lattice_d)**-3,\
                    labelstr='abel', colInt='blue', colRad='red' )
    axAb.set_ylabel('$n(r)$', rotation=0, fontsize=14, labelpad=5)

    # DECORATE AXES 
    if doabel:
        x00, x01 = axAz.get_xlim()
        x10, x11 = axAb.get_xlim()
        x0 = min(x00, x10)
        x1 = max(x01, x11)
        axAz.set_xticklabels([])
    else:
        x0, x1 = axAz.get_xlim()  
   
    for ax in axs:
        ax.set_xlim( x0, x1) 
        ax.grid(alpha=0.3,b=True) 
        if aznorm:
            ax.set_yticklabels(())
            ax.set_xticklabels(())
        #ax.tick_params(axis='both', which='major', labelsize=8)
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(6,prune='both'))
        ax.legend( loc='best', numpoints=1, prop={'size':6},\
                   handlelength=1.1, handletextpad=0.5 )
    return axAz


def Plot_CD_AZ( figure, gs0, **kwargs):
    azSimple = kwargs.get('azSimple', False) 
    if azSimple:
         gs = matplotlib.gridspec.GridSpecFromSubplotSpec(1,2,\
               subplot_spec=gs0, wspace=0.3, hspace=0.24,
               width_ratios=[1.6,1.])
    else:
         gs = matplotlib.gridspec.GridSpecFromSubplotSpec(1,3,\
               subplot_spec=gs0, wspace=0.3, hspace=0.24,
               width_ratios=[1.6,1.,1.])

    # PLOT THE COLUMN DENSITY 
    PlotCD( figure, gs[0,0], **kwargs) 

    # MAKE AZIMUTHAL, ABEL and COMPRESSIBILITY PLOTS 
    if azSimple:
        PlotAZsimple( figure, gsAz=gs[0,1], **kwargs)
    else:
        PlotAZ( figure, gsAz=gs[0,1], gsK=gs[0,2], **kwargs)


 
