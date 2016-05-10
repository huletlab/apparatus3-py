#!/usr/bin/python

import sys, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
sys.path.append('/lab/software/apparatus3/py')
import qrange, statdat, fitlibrary, coldens
from configobj import ConfigObj

import argparse

def average_cloud_centers( shots ):
    print shots 
    c0 = []; c1 = []; 
    for s in shots:
	report = ConfigObj( 'report%04d.INI'%s )  
        c0.append( qrange.evalstr( report, 'CPP:ax0c_crop' ) ) 
        c1.append( qrange.evalstr( report, 'CPP:ax1c_crop' ) )
    return np.mean( np.array(c0)), np.mean( np.array(c1)) 
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(__file__)
    parser.add_argument('--range', action = "store", \
           help="range of shots to be used.")
    parser.add_argument('--onlyCD', action= "store_true", \
           help="do only column density plot.")
    parser.add_argument('--onlyAZ', action= "store_true", \
           help="do only azimuthal average plot.")
    parser.add_argument('--onlyAZAB', action= "store_true", \
           help="do only azimuthal average and abel transform plot.")
    
    args = parser.parse_args()
    datakeys = ['SEQ:shot'] 
    data, errmsg, rawdat = qrange.qrange_eval( os.getcwd() , args.range, datakeys)
    print data
    print type(data)
    if not hasattr( data, '__iter__'):
    #if not isinstance( data, list):
        data = [ data ] 
    shots = [ int(i) for i in data ] 
    print shots
    # As an example run this in 140103
    # with --range 2959,2983,3016,3049 
  
   
    
    output = args.range 
    output = output.replace('-','m')
    output = output.replace(':','-')
    output = output.replace(',','_') 
    import datetime
    datestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_')
    savepath = 'plots/' 
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    outfile = savepath + 'densityInspec_' +  output + ".png" 
    
    
    # Get figure started
    from matplotlib import rc
    rc('font',**{'family':'serif'})
    rc('text', usetex=True)
  
    if args.onlyCD:
        figsize = (4.,4.) 
 
    elif args.onlyAZ: 
        azdict = {'azSimple':True}
        figsize = (8.,4.)
    elif args.onlyAZAB:
        azdict = {'azSimple':True, 'doabel':True}
        figsize = (8.,4.)
    else:
        azdict = {}
        figsize = (12., 4.)

    figure = plt.figure(figsize=figsize)
    #figure = plt.figure(figsize=(12.,2.5))
    gs0 = matplotlib.gridspec.GridSpec( 1,1, wspace=0.4, hspace=0.24,\
                                      top=0.90, left=0.01, right=0.97, bottom=0.15)
    
    
    # All the plots will be accomodated in one pane of a gridspec
    # this will facilitate putting together density profiles as a 
    # function of another variable 
    magnif = 1.497  # um/pixel
    lattice_d = 0.532 

    c0, r0 = average_cloud_centers( shots )
    r0 = int(r0) ; c0 = int(c0)
    if args.onlyCD:
        coldens.PlotCD( figure, gs0[0,0], dirpath=os.getcwd(), \
                            shots=shots, scale=True,\
                            magnif=magnif, \
                            lattice_d=lattice_d,\
                            cutPos='manual', r0=r0, c0=c0,)     

        cddata = coldens.average_cd( os.getcwd(), shots )
        print "\nSum of cddata = ", np.sum(cddata)
    else: 
 
        coldens.Plot_CD_AZ( figure, gs0[0,0], dirpath=os.getcwd(), \
                            shots=shots, 
                            title='column density', magnif=magnif, \
                            lattice_d=lattice_d,\
                            cutPos='manual', r0=r0, c0=c0, **azdict)     

        cddata = coldens.average_cd( os.getcwd(), shots )
        print "\nSum of cddata = ", np.sum(cddata)

        azdata = coldens.average_azcd( os.getcwd(), shots, trimStart=0)
        r = azdata[0]
        az = azdata[1]
        from scipy import integrate
        print "\nIntegral of azdata = ", 2*np.pi * integrate.simps( az*r, r) 
        
        from inverabel import inverAbelDat 
        abdat = inverAbelDat( azdata[0], azdata[1] )
        r = abdat[0][0] 
        ab = abdat[0][1] 
        print "\nIntegral of abdata = ", 4*np.pi * integrate.simps( ab*(r**2.), r)
     
 
    
    figure.savefig( outfile, dpi=250 ) 
    
