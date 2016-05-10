#!/usr/bin/python

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
sys.path.append('/lab/software/apparatus3/py')
import qrange, statdat, fitlibrary
from uncertainties import ufloat,unumpy

from scipy import stats

import argparse
parser = argparse.ArgumentParser('showbragg')
parser.add_argument('--range', action = "store", \
       help="range of shots to be used.")
parser.add_argument('--output', action="store", \
       help="optional path of png to save figure")
parser.add_argument('--image', action="store", type=float, default=285.5,\
       help="value of image for in-situ bragg data")
parser.add_argument('--imageTOF', action="store", type=float, default=285.5,\
       help="value of image for tof bragg data")


args = parser.parse_args()

tofval=0.006
braggdet = -117.
resdet = -152.3


savepath = 'plots/' 
if not os.path.exists(savepath):
    os.makedirs(savepath)

output = args.range 
output = output.replace('-','m')
output = output.replace(':','-')
output = output.replace(',','_') 

import datetime
datestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_')
outfile = savepath + "udata_" + datestamp + output + ".png" 
print outfile

gdat = {} 
gdat[ "udata_" + datestamp + output ] = {\
    'label':'udata',\
    'dir':os.getcwd(),\
    'shots':args.range,\
    'ec':'blue', 'fc':'lightblue',\
} 
  



datakeys = ['DIMPLELATTICE:knob05', 'SEQ:shot', 'ANDOR1EIGEN:signal' , 'ANDOR2EIGEN:signal', 'HHHEIGEN:andor2norm', 'DIMPLELATTICE:force_lcr3', 'DIMPLELATTICE:tof', 'DIMPLELATTICE:imgdet', 'DIMPLELATTICE:image' ]
def K( key ) : 
    return datakeys.index(key)

print "Fetching data..."
for k in gdat.keys():
    try:
        gdat[k]['data'] = np.loadtxt(k+'.dat')
        print "Loaded %s succesfully." % k 
    except:
        print k 
        data, errmsg, rawdat = qrange.qrange_eval( gdat[k]['dir'], gdat[k]['shots'], datakeys) 
        np.savetxt(k+'.data', data)
        gdat[k]['data'] = data

print "Done."
        

# Get figure started
from matplotlib import rc
rc('font',**{'family':'serif'})
figure = plt.figure(figsize=(12.,7.5))
gs = matplotlib.gridspec.GridSpec( 2,3, wspace=0.4, hspace=0.24,\
                                  top=0.90, left=0.1, right=0.97, bottom=0.1) 
figure.suptitle(r'U/t CURVE  (We use the shorthand $X_{t}\equiv\frac{X}{X_{\mathrm{TOF}}}$)')
ax1 = plt.subplot( gs[0,0] )
axA1 = plt.subplot( gs[0,1] )
axA2 = plt.subplot( gs[0,2] )
ax1T = plt.subplot( gs[1,0] )
axA1T = plt.subplot( gs[1,1] )
axA2T = plt.subplot( gs[1,2] )


base1=1.0
def fx(x):
    return x/100. * 3.82 

def plotkey( ax, gk, key, dat, base, marker='o', ms=5.,\
             mec='def', mfc='def', \
             labelstr='def',save=False, raw=True, raw_offset=0):
    if mec == 'def': mec = gdat[gk]['ec']
    if mfc == 'def': mfc = gdat[gk]['fc']
    if labelstr == 'def': labelstr = gdat[gk]['label']
    if raw:
        ax.plot( fx(dat[:, K('DIMPLELATTICE:knob05')])+raw_offset, dat[:,K(key)], '.',
                 marker='o', mec=mec, ms=3, mew=1.0, 
                 color='gray', alpha=0.5) 
     
    datS = statdat.statdat( dat, K('DIMPLELATTICE:knob05'), K(key) )
    alldat.append(datS) 
    ax.errorbar( fx(datS[:,0]), datS[:,1]/base, yerr=datS[:,3]/base,\
                  capsize=0., elinewidth=1.,\
                  fmt='.', ecolor=mec, mec=mec,\
                  mew=1.0, marker=marker, mfc=mfc, ms=ms,\
                  label=labelstr)
    if save:
        fname = key.replace(':','_') + '_' +  labelstr[-3:] + '.rawdat' 
        X =  np.transpose(np.vstack( ( fx(datS[:,0]), datS[:,1]/base, datS[:,3]/base )))
        np.savetxt( fname, X, fmt='%10.2f', delimiter='\t', newline='\n')
 
def plotkey_ratio( ax, gk, key, cond1, cond2, dat, base, marker='o', ms=5.,\
             mec='def', mfc='def', \
             labelstr='def', fit=False, save=False):
    if mec == 'def': mec = gdat[gk]['ec']
    if mfc == 'def': mfc = gdat[gk]['fc']
    if labelstr == 'def': labelstr = gdat[gk]['label'] 
   
    dat1 = np.copy(dat) 
    for c in cond1: dat1 = dat1[ dat1[:,K(c[0])] == c[1] ] 
    dat2 = np.copy(dat) 
    for c in cond2: dat2 = dat2[ dat2[:,K(c[0])] == c[1] ]

    plotkey = 'DIMPLELATTICE:knob05'

    x1= np.unique(dat1[:,K(plotkey)])
    x2= np.unique(dat2[:,K(plotkey)])

    set1 = set( x1.tolist())
    set2 = set( x2.tolist()) 
    common = list( set1 & set2 )
  
    num = dat1
    den = dat2
    rval = []
    rerr = []
    for c in common:
        numi = num[ num[:,K(plotkey)] == c ][ :, K(key) ] 
        deni = den[ den[:,K(plotkey)] == c ][ :, K(key) ] 
        val =  ufloat(( np.mean(numi), stats.sem(numi) )) / ufloat(( np.mean(deni), stats.sem(deni) ))
        rval.append( val.nominal_value )
        rerr.append( val.std_dev() )
    rval = np.array( rval ) / base 
    rerr = np.array( rerr ) / base
    xc = np.array(common)
    print '\t%s: x points in common = '%key,len(xc)
    if len(xc) > 0:
        ax.errorbar( fx(xc), rval/base, yerr=rerr/base,\
                  capsize=0., elinewidth=1.,\
                  fmt='.', ecolor=mec, mec=mec,\
                  mew=1.0, marker=marker, mfc=mfc, ms=ms,\
                  label=labelstr)
    if save:
        fname = key.replace(':','_') + '_' +  labelstr[-3:] + '.rawdatNOTCorrected' 
        X =  np.transpose(np.vstack( ( fx(xc), rval/base, rerr/base )))
        np.savetxt( fname, X, fmt='%10.2f', delimiter='\t', newline='\n')

    if fit:
        fitdat = np.transpose( np.vstack(( fx(xc), rval/base)))
        fun = fitlibrary.fitdict['Gaussian'].function
        p0 = [0.6, 0.01, 20., 1.]
        pFit , error = fitlibrary.fit_function( p0, fitdat,fun)
        fitX, fitY = fitlibrary.plot_function(pFit, fitdat[:,0],fun)
        #ax.plot( fitX, fitY, '-', color='gray', lw=4.5,zorder=1)
        print "1/e width Gaussian = ", pFit[2]

        fitdat = np.transpose( np.vstack(( fx(xc), rval/base-1.)))
        fun = fitlibrary.fitdict['GaussianNoOffset'].function
        p0 = [0.5, 0.0, 12.]
        pFit , error = fitlibrary.fit_function( p0, fitdat,fun)
        pFit = [0.6134, -3.027, 20.7434 ] #190a0 data
        #pFit = [0.5, 0.0, 12.] #400a0
        fitX, fitY = fitlibrary.plot_function(pFit, fitdat[:,0],fun)
        ax.plot( fitX, fitY+1., '-', color='gray', lw=6.5,zorder=1)
        print "1/e width GaussianNoOffset = ", pFit[2]
        fwhm = 2 * pFit[2] * np.sqrt( -1*np.log(1./2.) )
        print "FWHM = ", fwhm
        print "pFit = ", pFit
        ax.text( 0.05, 0.05, 'FWHM=%.0f mrad'%(fwhm*np.pi/180. *1000), transform=ax.transAxes,fontsize=10)





alltofdat = []
alldat=[]

for k in sorted(gdat.keys()):
    dat = gdat[k]['data']

    braggdet = -117.
    resdet = -152.3
    

    tofdat = dat[ dat[:,K('DIMPLELATTICE:tof')] == tofval ]
    tofdat = tofdat[ tofdat[:,K('DIMPLELATTICE:imgdet')] == braggdet ]
    tofdat = tofdat[ tofdat[:,K('DIMPLELATTICE:image')] == args.imageTOF ]
    print
    print "KEY = ",k
    print "TOFDATA @", np.unique(tofdat[:,K('DIMPLELATTICE:knob05')])
    #print "TOFDATA TOF", np.unique(tofdat[:,K('DIMPLELATTICE:tof')])
    #print "TOFDATA DET", np.unique(tofdat[:,K('DIMPLELATTICE:imgdet')])
   
    toflabel =  '$10\,\mu\mathrm{s}$ TOF'
    tof_offset = -0.8
    plotkey( ax1,k, 'HHHEIGEN:andor2norm', tofdat, base1, \
             marker='s', mec='black', mfc='None',  ms=4.,\
             labelstr=toflabel,\
             save=False, raw=True, raw_offset=tof_offset)
    plotkey( axA1,k, 'ANDOR1EIGEN:signal', tofdat, 1., \
             marker='s', mec='black', mfc='None', ms=4.,\
             labelstr=toflabel,\
             save=False, raw=True, raw_offset=tof_offset)

    plotkey( axA2,k, 'ANDOR2EIGEN:signal', tofdat, 1., \
             marker='s', mec='black', mfc='None', ms=4.,\
             labelstr=toflabel,\
             save=False, raw=True, raw_offset=tof_offset)

    alltofdat.append(tofdat[:,(K('DIMPLELATTICE:knob05'),K('HHHEIGEN:andor2norm'))])


    dat = dat[ dat[:,K('DIMPLELATTICE:force_lcr3')] == -1.] 
    dat = dat[ dat[:,K('DIMPLELATTICE:tof')] == 0. ]  
    dat = dat[ dat[:,K('DIMPLELATTICE:imgdet')] == braggdet ]
    dat = dat[ dat[:,K('DIMPLELATTICE:image')] == args.image ]
    print "DATA @", np.unique(dat[:,K('DIMPLELATTICE:knob05')])
    #print "DATA TOF", np.unique(dat[:,K('DIMPLELATTICE:tof')])
    #print "DATA DET", np.unique(dat[:,K('DIMPLELATTICE:imgdet')])

    insitu_offset = 0.8
    plotkey( ax1,k, 'HHHEIGEN:andor2norm', dat, base1, labelstr='In-situ',save=False,
             raw=True, raw_offset=insitu_offset) 
    plotkey( axA1,k, 'ANDOR1EIGEN:signal', dat, 1., labelstr='In-situ',save=False,
             raw=True, raw_offset=insitu_offset) 
    plotkey( axA2,k, 'ANDOR2EIGEN:signal', dat, 1., labelstr='In-situ',save=False,
             raw=True, raw_offset=insitu_offset) 

    plotkey_ratio( ax1T,k, 'HHHEIGEN:andor2norm',\
                   [('DIMPLELATTICE:tof',0.0),('DIMPLELATTICE:imgdet',braggdet),
                    ('DIMPLELATTICE:image',args.image) ],\
                   [('DIMPLELATTICE:tof',tofval),('DIMPLELATTICE:imgdet',braggdet),
                    ('DIMPLELATTICE:image',args.imageTOF)],\
                   gdat[k]['data'], 1.0, fit=False)

    plotkey_ratio( axA1T,k, 'ANDOR1EIGEN:signal',\
                   [('DIMPLELATTICE:tof',0.0),('DIMPLELATTICE:imgdet',braggdet),
                    ('DIMPLELATTICE:image',args.image)],\
                   [('DIMPLELATTICE:tof',tofval),('DIMPLELATTICE:imgdet',braggdet),
                    ('DIMPLELATTICE:image',args.imageTOF)],\
                   gdat[k]['data'], 1.0, save=False)

    plotkey_ratio( axA2T,k, 'ANDOR2EIGEN:signal',\
                   [('DIMPLELATTICE:tof',0.0),('DIMPLELATTICE:imgdet',braggdet),
                    ('DIMPLELATTICE:image',args.image)],\
                   [('DIMPLELATTICE:tof',tofval),('DIMPLELATTICE:imgdet',braggdet),
                    ('DIMPLELATTICE:image',args.imageTOF)],\
                   gdat[k]['data'], 1.0, save=False)

    tofset = set( np.unique(tofdat[:,K('DIMPLELATTICE:knob05')]).tolist() )
    datset = set( np.unique(dat[:,K('DIMPLELATTICE:knob05')]).tolist() )
    common = list( tofset & datset )

    np.set_printoptions(suppress=True, precision=3)
    for i,c in enumerate(sorted(common)):
        tofi = tofdat[ tofdat[:,K('DIMPLELATTICE:knob05')] == c ]
        dati = dat[ dat[:,K('DIMPLELATTICE:knob05')] == c ]
        print '\nKNOB5 = ', c 
    
        #print 
        #knob05 = 200. 
        cols = ( K('SEQ:shot'), K('ANDOR1EIGEN:signal'), K('ANDOR2EIGEN:signal'), \
                 K('HHHEIGEN:andor2norm'), K('DIMPLELATTICE:tof'), K('DIMPLELATTICE:image') ) 
        print "DAT"
        print dati[:,cols]
        print "TOFDAT"
        print tofi[:,cols]


alltofdat = np.vstack(alltofdat)
am = np.mean(alltofdat[:,1])
a0 = np.mean(alltofdat[:,1]) - stats.sem(alltofdat[:,1])
a1 = np.mean(alltofdat[:,1]) + stats.sem(alltofdat[:,1])

ax1.set_ylabel(r'$\frac{A2}{A1}$',ha='center',labelpad=20, rotation=0,fontsize=22)
axA1.set_ylabel(r'$A1$', ha='center', labelpad=20, rotation=0, fontsize=18)
axA2.set_ylabel(r'$A2$', ha='center', labelpad=20, rotation=0, fontsize=18)

ax1T.set_ylabel(r'$\left(\frac{  A2 }{  A1 }\right)_{t}$',\
                ha='center',labelpad=30, rotation=0,fontsize=22)
axA1T.set_ylabel(r'$A1_{t}$', ha='center', labelpad=20, rotation=0, fontsize=18)
axA2T.set_ylabel(r'$A2_{t}$', ha='center', labelpad=20, rotation=0, fontsize=18)


axes = [ ax1, axA1, axA2, ax1T, axA1T, axA2T ]
#axes = [ ax1T, axA1T, axA2T ]
for ax in axes:
    ax.grid()
    ax.set_xlabel('$U/t$')

    #for l in ax.xaxis.get_ticklabels():
    #    l.set_rotation(30)
    #ax.xaxis.set_major_formatter(\
    #     matplotlib.ticker.FormatStrFormatter( '%d$^{\circ}$' ) )
    #ax1.xaxis.set_major_locator( matplotlib.ticker.MultipleLocator(0.1) )

    ax.legend(loc='best',numpoints=1,prop={'size':8},\
               handlelength=1.1,handletextpad=0.5)

#axA2T.set_ylim(0.8,1.6)
#axA1T.set_ylim(0.7,1.1)
#ax1T.set_ylim(0.8,1.8)



for pos in ['top','bottom','right','left']:
    ax1.spines[pos].set_edgecolor('green') 
    axA1.spines[pos].set_edgecolor('green') 
    axA2.spines[pos].set_edgecolor('green') 
    ax1.spines[pos].set_linewidth(2.0) 
    axA1.spines[pos].set_linewidth(2.0) 
    axA2.spines[pos].set_linewidth(2.0) 
#    ax.xaxis.set_ticks( sorted(A2ticks.keys()) )
#    ax.xaxis.set_ticklabels( [ Qv(A2ticks[k]) for k in sorted(A2ticks.keys()) ] ) 
#    for l in ax.xaxis.get_ticklabels():
#        l.set_rotation(30)
#        l.set_fontsize(8)

#axes = [ax1, axA1, axA2]
#for ax in axes:
#    ax.text( 0.01, 0.01, '*Gray squares are tof=%.2f ms'%tofval, \
#             transform=ax.transAxes, fontsize=12, color='red')


#gs.tight_layout(figure, rect=[0,0.0,1.,0.95])
plt.savefig(outfile, dpi=250)

exit()

