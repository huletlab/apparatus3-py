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

magnif = 1.497 / 1.03228 # um per pixel
lattice_d = 0.532



# NEED TO ATTEMPT A CORRECTION OF THE AZ-AVERAGE ASPECT RATIO ADJUSTMENT
# WHICH IS POSSIBLY WHAT IS AFFECTING THE DENSITY MAKING IT APPEAR
# SYSTEMATICALLY LOWER. 
 
#=============================================================================
from scipy import integrate
def integrate_sphere( r, qty):
    q = qty[ ~np.isnan(qty)]
    r = r[ ~np.isnan(qty) ]
    a = 1.064/2.
    return np.power( a,-3) * 4*np.pi * integrate.simps( q*(r**2), r)

def integrate_circle( r, qty):
    q = qty[ ~np.isnan(qty)]
    r = r[ ~np.isnan(qty) ]
    a = 1.064/2.
    return np.power( a,-2) * 2*np.pi * integrate.simps( q*r, r)

#=============================================================================

from configobj import ConfigObj

def abel_single( datadir, shot, **kwargs ):

    # Find out the aspect ratio 
    inifile = datadir + 'report' + "%04d"%shot + '.INI'
    report = ConfigObj(inifile) 
    try:
        image   = float( report['DIMPLELATTICE']['image'])
        num     = float( report['CPP']['nfit_mott'] )
        ncount  = float( report['CPP']['ncount'] )
        ax0w    = float( report['CPP']['ax0w']) 
        ax1w    = float( report['CPP']['ax1w']) 
        AR = ax1w / ax0w
        #print "AR = ", AR
        #AR = 1.0 
    
        ir = float( report['DIMPLE']['ir1pow0'] )  
        aS = float( report['DIMPLELATTICE']['knob05'] ) 
        gr = float( report['DIMPLELATTICE']['knob01'] )
        v0 = float( report['DIMPLELATTICE']['knob23'] )
        dthold = float( report['DIMPLELATTICE']['dthold'] )  
     
        peakd_sph  = float( report['CPP']['peakd_sph'] )
        peakd_mott = float( report['CPP']['peakd_mott'] )
      
        r0_mott = float( report['CPP']['r0_mott'] ) 
        ax0w_mott = float( report['CPP']['ax0w_mott'] ) 
        ax1w_mott = float( report['CPP']['ax1w_mott'] )

  
    except Exception as e:
        print "ERROR getting value from INI file.  shot# = ", shot
        raise e 


    try:
        gr1c = float( report['DIMPLE']['gr1correct'] ) 
        gr2c = float( report['DIMPLE']['gr2correct'] ) 
        gr3c = float( report['DIMPLE']['gr3correct'] ) 
    except:
        gr1c = 1. 
        gr2c = 1. 
        gr3c = 1. 
        
    # Columns for data file will be 
    #  0 - shot 
    #  1 - nfit_mott
    #  2 - ax0w  (um)
    #  3 - ax1w  (um) 
    #  4 - AR 
    #  5 - ir 
    #  6 - aS 
    #  7 - gr 
    #  8 - ncount 
    #  9 - peakd_sph
    # 10 - peakd_mott
    # 11 - gr1correct
    # 12 - gr2correct
    # 13 - gr3correct
    # 14 - image
    # 15 - r0_mott
    # 16 - ax0w_mott
    # 17 - ax1w_mott  
    # 18 - v0
    # 19 - dthold 
    
    line = [shot , num, ax0w, ax1w, AR, ir, aS, gr,  \
               ncount, peakd_sph, peakd_mott, gr1c, gr2c, gr3c, image, r0_mott,\
               ax0w_mott, ax1w_mott, v0, dthold]
    if kwargs.get('printline', False):
        print line 

    fmt = '%4d' + '%10.4g' + '%7.2f'*6 +  '%10.4g'*3 + '%6.3f'*3 + '%7.1f' \
          + '%7.2f'*5
            

    recordsfile = kwargs.get('recordsfile', None) 
    skipdone = kwargs.get('skipdone', False)
  
    if recordsfile is not None:
        try:
            if skipdone:
                records = np.loadtxt(recordsfile)  
                if shot in records[:,0].tolist():
                    print "skipping #{:04d}".format(int(shot))
                    return
        except:
            pass 
       
        recordsdir = os.path.dirname( recordsfile )  
        if not os.path.exists( recordsdir ) :
            os.makedirs( recordsdir )  
    
        with open( recordsfile, 'a' ) as f:
            np.savetxt( f, np.atleast_2d(line), fmt=fmt ) 


    figTest = Figure( figsize=(10., 3.5 ) )
    canvas = FigureCanvas( figTest ) 
    gsTest = matplotlib.gridspec.GridSpec( 1,3, 
                wspace=0.35, hspace=0.15,\
                left=0.03, right=0.97, bottom=0.10, top=0.92 ) 
    
    cddata = np.loadtxt( os.path.join( datadir, '%04d_column.ascii'%shot ) ) 
    azdata = coldens.average_azcd( datadir, [shot] ) 
    
    coldens.PlotCD(figTest, gsTest[0,0], \
        cddata = cddata,\
        dirpath=datadir,\
        title=None)
    
    color = 'blue' 
    label = None
    axaz = figTest.add_subplot( gsTest[0,1] ) 
    azRad, azDens = coldens.azplot( axaz, azdata,\
                    xs=magnif, ys=(magnif/lattice_d)**-2,\
                    labelstr=label, col=color )

 
    axaz.grid(alpha=0.3) 
    axaz.set_ylim(0., 100.)
    axaz.set_xlim(0., 40. ) 
   
    
    #print azdata
    numberAZ = integrate_circle( azRad, azDens ) 
    #print "Number from integral of AZ :", numberAZ
    #print azdata
    
    azdata = ( azdata[0]/AR, azdata[1] ) 
    
    axab = figTest.add_subplot( gsTest[0,2] )  

    xscale = magnif*np.sqrt(AR)
    yscale = (magnif/lattice_d)**-3 
    abDat = coldens.azabel( axab, None, azdata,\
                    trimStart=0,\
                    xs=xscale, ys=yscale,\
                    labelstr=label, col=color )

    # Optional - rescale the azdata before it goes to file
    # azdata = ( azdata[0] * xscale, azdata[1] * ((magnif/lattice_d)**-2) )

    axab.grid(alpha=0.3) 
    axab.set_ylim(-0.1, 2.1)
    axab.set_xlim(0., 30. ) 
    
    #print abDat[0]  
    numberAB = integrate_sphere( abDat[0][:,0] , abDat[0][:,1] )
    #print "Number from integral of AB :", numberAB

    axaz.text( 1.02, 1.03,  r'$U/t={:0.2f}$'.format( aS * 11. / 290. ) + '\n'+\
         r'$\mathrm{{ir}}={:0.2f}$'.format( ir) + '\n' +\
         r'$\mathrm{{AR}}={:0.2f}$'.format( AR ) + '\n' +\
         r'$N={:0.2f}\times 10^{{5}}$'.format( num/1e5 )  + '\n' +\
         r'$N_{{AZ}}={:0.2f}\times 10^{{5}}$'.format( numberAZ/1e5 )  + '\n' +\
         r'$N_{{AB}}={:0.2f}\times 10^{{5}}$'.format( numberAB/1e5 ) ,
               ha='right', va='top', transform=axaz.transAxes,\
               bbox = {'boxstyle':'round', 'facecolor':'white'} ) 

    #savedir = 'abelsingle/{:03d}/{:.2f}/'.format( int(aS), gr ) 
    savedir = 'plots/'
    if not os.path.exists( savedir ) :
        os.makedirs( savedir )  

    base = 'ABEL_ir{:0.2f}_image{:0.1f}_num{:0.2f}_shot{:04d}'.\
             format( ir , image, num/1e5, int(shot) ) 
  
    fname = base + '.png' 
    canvas.print_figure( savedir + fname , dpi=250)
    # plt is evil, leaks memory.  switched to Figure and canvas approach
    #plt.close(figTest)
  
    # Save abel transformed data 
    datfname = base + '_abel.dat' 
    np.savetxt( savedir + datfname, abDat[0], fmt='%.6g' )

    # Save the raw column density 
    cddatfname = base + '_cd.dat' 
    np.savetxt( savedir + cddatfname, cddata, fmt='%.6g'  )

    # Save the raw azimuthal average data 
    azdatfname = base + '_az.dat' 
    np.savetxt( savedir + azdatfname, np.column_stack( azdata ), fmt='%.6g' )
   
    
    

if __name__ == '__main__':
#    datadir = '/lab/data/app3/2015/1506/150617/'
#    shot = 5282

	parser = argparse.ArgumentParser()
	parser.add_argument("shot",help="shot to do abel")
	args = parser.parse_args()
	shot = int(args.shot)
	datadir = "./"
	abel_single( datadir, shot, printline=True ) 



