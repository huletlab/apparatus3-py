#!/usr/bin/python

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
sys.path.append('/lab/software/apparatus3/py')
import qrange, statdat, fitlibrary
from uncertainties import ufloat,unumpy
from configobj import ConfigObj
from scipy import stats

import argparse
parser = argparse.ArgumentParser('showbragg')
parser.add_argument('--range', action = "store", \
       help="range of shots to be used.")
parser.add_argument('--output', action="store", \
       help="optional path of png to save figure")
parser.add_argument('--image', action="store", type=float, default=285.5,\
       help="value of image for in-situ bragg data")
parser.add_argument('--imageTOFlock', action="store", type=float, default=251.0,\
       help="value of image for locked tof bragg data")
parser.add_argument('--imageTOFassoc', action="store", type=float, default=285.5,\
       help="value of image for assoc tof bragg data")
parser.add_argument('--tofval', action="store", type=float, default=0.006,\
       help="value of DL.tof that represents a TOF shot")
parser.add_argument('--braggdet', action="store",type=float, default=-117.,\
       help="value of detuning for Bragg shots")
parser.add_argument('--latticedepth', action="store",type=float, default=5.5,\
       help="depth of hte lattice in Er")
parser.add_argument('--binning', action="store",type=float, default=500,\
       help="bin for the andor extra data")
parser.add_argument('--binning_scale', action="store",type=float, default=1.0,\
       help="scale of the bin data")
parser.add_argument('--binning_center', action="store",type=float, default=0.0,\
       help="center of the bin data")
parser.add_argument('--varyimage', action="store_true", default=False,\
       help="use this if image key is varying throught set")

aSkey = 'DIMPLELATTICE:knob05' 
parser.add_argument('--xkey', action="store", type=str, default=aSkey,\
       help="name of the report key for the X axis")


args = parser.parse_args()


savepath = 'plots/' 
if not os.path.exists(savepath):
    os.makedirs(savepath)

output = args.range 
output = output.replace('-','m')
output = output.replace(':','-')
output = output.replace(',','_') 

import datetime
datestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_')

if args.xkey != aSkey:
    xkeystr = "_%s_"%args.xkey
    xkeystr = xkeystr.replace(':','_') 
else:
    xkeystr = '' 
outfile = savepath + "udata_ernie_" + datestamp + output + xkeystr + ".png" 
print outfile


from DataHandling import data_fetch, data_ratio, data_pick, plotkey, plotkey_ratio 



def roundUpToMultiple(number, multiple,center=0):
	if center ==0:	
  		num = number + (multiple - 1)
  		return num - (num % multiple)
	else:
		start = center - multiple/2
		number = number-start
  		num = number + (multiple - 1)
  		return num - (num % multiple)+start

extraKeys=["ANDOR1EXTRA","ANDOR2EXTRA"]

if args.xkey.split(":")[0] in extraKeys:
	print "Processing the shots to find nearest TOF shot"
	matchKey = 'DIMPLE:ir1pow0'
	datakeys2 = [matchKey, 'SEQ:shot', 'ANDOR1EIGEN:signal',\
            'ANDOR2EIGEN:signal', 'HHHEIGEN:andor2norm',\
            'DIMPLELATTICE:force_lcr3', 'DIMPLELATTICE:tof',\
            'DIMPLELATTICE:imgdet', 'DIMPLELATTICE:image' ]
	gdat2 = {} 
	gdat2[ "udata_" + datestamp + output ] = {\
	    'label':'udata',\
	    'dir':os.getcwd(),\
	    'shots':args.range,\
    		'ec':'blue', 'fc':'blue',\
	      } 
	gdat, K  = data_fetch( datakeys2, gdat2, save=False) 
	for k in sorted(gdat.keys()):
	    dat = gdat[k]['data']
	    tofLock_cond = [ ('DIMPLELATTICE:tof', args.tofval),\
	                     ('DIMPLELATTICE:imgdet', args.braggdet)]
	    if not args.varyimage:
	        tofLock_cond = tofLock_cond +  [('DIMPLELATTICE:image', args.imageTOFlock)]
	
	    inSitu_cond = [('DIMPLELATTICE:tof',0.0),('DIMPLELATTICE:imgdet',args.braggdet),
	                    ('DIMPLELATTICE:force_lcr3', -1) ]
	
	    if not args.varyimage:
	        inSitu_cond = inSitu_cond + [('DIMPLELATTICE:image',args.image)]
	
	    tofLock  = data_pick( dat, tofLock_cond , K )
	    inSitu   = data_pick( dat, inSitu_cond, K )
	    print "LOCKTOF DATA  @", np.unique(tofLock[:,K(matchKey)])
	    print "INSITU DATA   @", np.unique(inSitu[:,K(matchKey)])
	
	    tofLockset = set( np.unique(tofLock[:,K(matchKey)]).tolist() )
	    inSituset = set( np.unique(inSitu[:,K(matchKey)]).tolist() )
	    common = list( tofLockset & inSituset )
	    print common
	    np.set_printoptions(suppress=True, precision=3)
	    for i,c in enumerate(sorted(common)):
	        tofLocki = tofLock[ tofLock[:,K(matchKey)] == c ]
	        inSitui = inSitu[ inSitu[:,K(matchKey)] == c ]
	        for n,j in enumerate(inSitui):
	                index, shot = min(enumerate(tofLocki[:,K("SEQ:shot")]),key=lambda l: abs(l[1]-j[K("SEQ:shot")]))
	                ishot = "report%04d.INI" %j[K('SEQ:shot')]
	                tshot = "report%04d.INI" %shot
	                ireport = ConfigObj(ishot)
	                treport = ConfigObj(tshot)
	                ireport["ANDOR1EXTRA"]={}
	                ireport["ANDOR2EXTRA"]={}
	                treport["ANDOR1EXTRA"]={}
	                treport["ANDOR2EXTRA"]={}
	                ireport["ANDOR1EXTRA"]["nearestTof_signal"]=tofLocki[index,K('ANDOR1EIGEN:signal')]
	                ireport["ANDOR1EXTRA"]["nearestTof_signal_bin"]=roundUpToMultiple(tofLocki[index,K('ANDOR1EIGEN:signal')],args.binning,args.binning_center)*args.binning_scale
	                ireport["ANDOR1EXTRA"]["nearestTof_bin"]=args.binning
	                ireport["ANDOR1EXTRA"]["nearestTof_shot"]=shot
	                ireport["ANDOR2EXTRA"]["nearestTof_signal"]=tofLocki[index,K('ANDOR2EIGEN:signal')]
	                ireport["ANDOR2EXTRA"]["nearestTof_signal_bin"]=roundUpToMultiple(tofLocki[index,K('ANDOR2EIGEN:signal')],args.binning,args.binning_center)*args.binning_scale
	                ireport["ANDOR2EXTRA"]["nearestTof_bin"]=args.binning
	                ireport["ANDOR2EXTRA"]["nearestTof_shot"]=shot
	                treport["ANDOR1EXTRA"]["nearestTof_signal"]=tofLocki[index,K('ANDOR1EIGEN:signal')]
	                treport["ANDOR1EXTRA"]["nearestTof_signal_bin"]=roundUpToMultiple(tofLocki[index,K('ANDOR1EIGEN:signal')],args.binning,args.binning_center)*args.binning_scale
	                treport["ANDOR1EXTRA"]["nearestTof_bin"]=args.binning
	                treport["ANDOR1EXTRA"]["nearestTof_shot"]=shot
	                treport["ANDOR2EXTRA"]["nearestTof_signal"]=tofLocki[index,K('ANDOR2EIGEN:signal')]
	                treport["ANDOR2EXTRA"]["nearestTof_signal_bin"]=roundUpToMultiple(tofLocki[index,K('ANDOR2EIGEN:signal')],args.binning,args.binning_center)*args.binning_scale
	                treport["ANDOR2EXTRA"]["nearestTof_bin"]=args.binning
	                treport["ANDOR2EXTRA"]["nearestTof_shot"]=shot
	                ireport.write()
	                treport.write()


gdat = {} 
gdat[ "udata_" + datestamp + output ] = {\
    'label':'udata',\
    'dir':os.getcwd(),\
    'shots':args.range,\
    'ec':'blue', 'fc':'blue',\
} 
  
datakeys = [args.xkey, 'SEQ:shot', 'ANDOR1EIGEN:signal',\
            'ANDOR2EIGEN:signal', 'HHHEIGEN:andor2norm',\
            'DIMPLELATTICE:force_lcr3', 'DIMPLELATTICE:tof',\
            'DIMPLELATTICE:imgdet', 'DIMPLELATTICE:image' ]

gdat, K  = data_fetch( datakeys, gdat, save=False) 



# Get figure started
from matplotlib import rc
rc('font',**{'family':'serif'})
figure = plt.figure(figsize=(20.,7.5))
gs = matplotlib.gridspec.GridSpec( 2,4, wspace=0.4, hspace=0.24,\
                                  top=0.90, left=0.07, right=0.97, bottom=0.1) 
figure.suptitle(r'U/t CURVE  (We use the shorthand $X_{t}\equiv\frac{X}{X_{\mathrm{TOF}}}$)')
ax1 = plt.subplot( gs[0,0] )
axA1 = plt.subplot( gs[0,1] )
axA2 = plt.subplot( gs[0,2] )
ax1T = plt.subplot( gs[1,0] )
axA1T = plt.subplot( gs[1,1] )
axA2T = plt.subplot( gs[1,2] )

axS1 = plt.subplot( gs[0,3] ) 
axS2 = plt.subplot( gs[1,3] ) 

base1=1.0
def fx(x):
    if args.xkey == 'DIMPLELATTICE:knob05': 
        if args.latticedepth == 5.5:  
            wF = 11.866 
            t  = 0.0577 
        elif args.latticedepth == 7.0:
            wF = 15.1877
            t  = 0.0394 
        a0a = 5.29e-11 / (1064e-9/2.) 
        U = x * a0a*wF
        return U/t
    else:
        return x 


for k in sorted(gdat.keys()):
    dat = gdat[k]['data']
    tofLock_cond = [ ('DIMPLELATTICE:tof', args.tofval),\
                     ('DIMPLELATTICE:imgdet', args.braggdet)]
    if not args.varyimage:
        tofLock_cond = tofLock_cond +  [('DIMPLELATTICE:image', args.imageTOFlock)]

    tofAssoc_cond = [ ('DIMPLELATTICE:tof', args.tofval),\
                     ('DIMPLELATTICE:imgdet', args.braggdet),\
                     ('DIMPLELATTICE:image', args.imageTOFassoc) ]

    inSitu_cond = [('DIMPLELATTICE:tof',0.0),('DIMPLELATTICE:imgdet',args.braggdet),
                    ('DIMPLELATTICE:force_lcr3', -1) ]
    if not args.varyimage:
        inSitu_cond = inSitu_cond + [('DIMPLELATTICE:image',args.image)]

    tofLock  = data_pick( dat, tofLock_cond , K ) 
    tofAssoc = data_pick( dat, tofAssoc_cond, K ) 
    inSitu   = data_pick( dat, inSitu_cond, K )

    print "LOCKTOF DATA  @", np.unique(tofLock[:,K(args.xkey)])
    print "ASSOCTOF DATA @", np.unique(tofAssoc[:,K(args.xkey)])
    print "INSITU DATA   @", np.unique(inSitu[:,K(args.xkey)])
 
    # PLOT ALL THE tofLock DATA 
    tofLock_label =  '$10\,\mu\mathrm{s}$ TOF'
    tofLock_offset = -0.4
    tofLock_color = 'black' 
    plotkey( ax1, gdat[k], K, fx, args.xkey, 'HHHEIGEN:andor2norm', tofLock, base1, \
             marker='s', mec=tofLock_color, mfc='None',  ms=4.,\
             labelstr=tofLock_label,\
             save=False, raw=True, raw_offset=tofLock_offset)
    plotkey( axA1, gdat[k], K, fx, args.xkey, 'ANDOR1EIGEN:signal', tofLock, 1., \
             marker='s', mec=tofLock_color, mfc='None', ms=4.,\
             labelstr=tofLock_label,\
             save=False, raw=True, raw_offset=tofLock_offset)
    plotkey( axA2,gdat[k], K, fx, args.xkey, 'ANDOR2EIGEN:signal', tofLock, 1., \
             marker='s', mec=tofLock_color, mfc='None', ms=4.,\
             labelstr=tofLock_label,\
             save=False, raw=True, raw_offset=tofLock_offset)

    # PLOT ALL THE tofAssoc DATA 
    tofAssoc_label =  '$10\,\mu\mathrm{s}$ TOF - Assoc'
    tofAssoc_offset = -0.6
    tofAssoc_color = 'red' 
    plotkey( ax1, gdat[k], K, fx, args.xkey, 'HHHEIGEN:andor2norm', tofAssoc, base1, \
             marker='s', mec=tofAssoc_color, mfc='None',  ms=4.,\
             labelstr=tofAssoc_label,\
             save=False, raw=True, raw_offset=tofAssoc_offset)
    plotkey( axA1, gdat[k], K, fx, args.xkey, 'ANDOR1EIGEN:signal', tofAssoc, 1., \
             marker='s', mec=tofAssoc_color, mfc='None', ms=4.,\
             labelstr=tofAssoc_label,\
             save=False, raw=True, raw_offset=tofAssoc_offset)
    plotkey( axA2, gdat[k], K, fx, args.xkey, 'ANDOR2EIGEN:signal', tofAssoc, 1., \
             marker='s', mec=tofAssoc_color, mfc='None', ms=4.,\
             labelstr=tofAssoc_label,\
             save=False, raw=True, raw_offset=tofAssoc_offset)

    # PLOT ALL THE inSitu DATA 
    insitu_offset = 0.4
    plotkey( ax1, gdat[k], K, fx, args.xkey, 'HHHEIGEN:andor2norm',\
             inSitu, base1, labelstr='In-situ',save=False,
             raw=True, raw_offset=insitu_offset) 
    plotkey( axA1, gdat[k], K, fx, args.xkey, 'ANDOR1EIGEN:signal',\
              inSitu, 1., labelstr='In-situ',save=False,
             raw=True, raw_offset=insitu_offset) 
    plotkey( axA2, gdat[k], K, fx, args.xkey, 'ANDOR2EIGEN:signal',\
             inSitu, 1., labelstr='In-situ',save=False,
             raw=True, raw_offset=insitu_offset)

    ############
    #  RATIOS  
    ############
    
    # tofLock ratio
    plotkey_ratio( ax1T, gdat[k], K, fx, args.xkey, 'HHHEIGEN:andor2norm',\
                   inSitu_cond, tofLock_cond, gdat[k]['data'], 1.0, labelstr='TOF', exceptions=True)
    plotkey_ratio( axA1T, gdat[k], K, fx, args.xkey, 'ANDOR1EIGEN:signal',\
                   inSitu_cond, tofLock_cond, gdat[k]['data'], 1.0, save=False, labelstr='TOF')
    plotkey_ratio( axA2T, gdat[k], K, fx, args.xkey, 'ANDOR2EIGEN:signal',\
                   inSitu_cond, tofLock_cond, gdat[k]['data'], 1.0, save=False, labelstr='TOF')

    # tofAssoc ratio
    plotkey_ratio( ax1T, gdat[k], K, fx, args.xkey, 'HHHEIGEN:andor2norm',\
                   inSitu_cond, tofAssoc_cond, gdat[k]['data'], 1.0, \
                   mec=tofAssoc_color, mfc=tofAssoc_color, labelstr='TOF-Assoc')
    plotkey_ratio( axA1T, gdat[k], K, fx, args.xkey, 'ANDOR1EIGEN:signal',\
                   inSitu_cond, tofAssoc_cond, gdat[k]['data'], 1.0, \
                   mec=tofAssoc_color, mfc=tofAssoc_color, save=False, labelstr='TOF-Assoc')
    plotkey_ratio( axA2T, gdat[k], K, fx, args.xkey, 'ANDOR2EIGEN:signal',\
                   inSitu_cond, tofAssoc_cond, gdat[k]['data'], 1.0, \
                   mec=tofAssoc_color, mfc=tofAssoc_color, save=False, labelstr='TOF-Assoc')

    ###############################
    #  CORRECTED FOR DW AND Isat 
    ##############################

    def SQ2( It ):
        DW = 0.81 
        s0 = 15. 
        Detuning = 6.5 
        return 1 + (It - 1 ) * (1+ s0/(4.*(Detuning**2))) / DW 

    def SQ1( It ):
        DW = 0.95
        s0 = 15. 
        Detuning = 6.5 
        return 1 + (It - 1 ) * (1+ s0/(4.*(Detuning**2))) / DW 
    
    # tofLock ratio
    plotkey_ratio( axS1, gdat[k], K, fx, args.xkey, 'ANDOR1EIGEN:signal',\
                   inSitu_cond, tofLock_cond, gdat[k]['data'], 1.0, \
                   save=False, labelstr='TOF', yf=SQ1)
    plotkey_ratio( axS2, gdat[k], K, fx, args.xkey, 'ANDOR2EIGEN:signal',\
                   inSitu_cond, tofLock_cond, gdat[k]['data'], 1.0, \
                   save=False, labelstr='TOF', yf=SQ2)

    # tofAssoc ratio
    plotkey_ratio( axS1, gdat[k], K, fx, args.xkey, 'ANDOR1EIGEN:signal',\
                   inSitu_cond, tofAssoc_cond, gdat[k]['data'], 1.0, \
                   mec=tofAssoc_color, mfc=tofAssoc_color, \
                   save=False, labelstr='TOF-Assoc', yf=SQ1)
    plotkey_ratio( axS2, gdat[k], K, fx, args.xkey, 'ANDOR2EIGEN:signal',\
                   inSitu_cond, tofAssoc_cond, gdat[k]['data'], 1.0, \
                   mec=tofAssoc_color, mfc=tofAssoc_color,\
                   save=False, labelstr='TOF-Assoc', yf=SQ2)


    tofLockset = set( np.unique(tofLock[:,K(args.xkey)]).tolist() )
    tofAssocset = set( np.unique(tofAssoc[:,K(args.xkey)]).tolist() )
    inSituset = set( np.unique(inSitu[:,K(args.xkey)]).tolist() )
    common = list( tofLockset & tofAssocset & inSituset  )

    if len(tofAssocset) == 0: 
        common = list( tofLockset & inSituset ) 
    np.set_printoptions(suppress=True, precision=3)
    for i,c in enumerate(sorted(common)):
        print "inside the print loop"
	tofLocki = tofLock[ tofLock[:,K(args.xkey)] == c ]
        inSitui = inSitu[ inSitu[:,K(args.xkey)] == c ]
	#### Start to label them with the andor 2 count ###
	temp =[]
	temp2=[]
	for n,i in enumerate(inSitui):
		index, shot = min(enumerate(tofLocki[:,1]),key=lambda j: abs(j[1]-i[1]))
		temp.append(np.append(i,[shot,tofLocki[index,K('ANDOR1EIGEN:signal')]]))
		for l in gdat[k]['data']:
			if l[K("SEQ:shot")]==i[K('SEQ:shot')]:
				l[-1]=roundUpToMultiple(tofLocki[index,K('ANDOR1EIGEN:signal')],args.binning,args.binning_center)*args.binning_scale
				break
		for l in gdat[k]['data']:
			if l[K("SEQ:shot")]==shot:
				l[-1]=roundUpToMultiple(tofLocki[index,K('ANDOR1EIGEN:signal')],args.binning,args.binning_center)*args.binning_scale
				break
        print '\nKNOB5 = ', c 
        #print 
        #knob05 = 200. 
        cols = ( K('SEQ:shot'), K('ANDOR1EIGEN:signal'), K('ANDOR2EIGEN:signal'), \
                 K('HHHEIGEN:andor2norm'), K('DIMPLELATTICE:tof'), K('DIMPLELATTICE:image') ,-1,-2) 
        print "INSITU DAT "
        print inSitui[:,cols]
        print "TOF LOCK DAT"
        print tofLocki[:,cols]
        if len(tofAssocset) > 0 : 
            tofAssoci = tofAssoc[ tofAssoc[:,K(args.xkey)] == c ]
            print "TOF ASSOC DAT"
            print tofAssoci[:,cols]

# Y labels for all axes 
ax1.set_ylabel(r'$\frac{A2}{A1}$',ha='center',labelpad=20, rotation=0,fontsize=22)
axA1.set_ylabel(r'$A1$', ha='center', labelpad=20, rotation=0, fontsize=18)
axA2.set_ylabel(r'$A2$', ha='center', labelpad=20, rotation=0, fontsize=18)

ax1T.set_ylabel(r'$\left(\frac{  A2 }{  A1 }\right)_{t}$',\
                ha='center',labelpad=30, rotation=0,fontsize=22)
axA1T.set_ylabel(r'$A1_{t}$', ha='center', labelpad=20, rotation=0, fontsize=18)
axA2T.set_ylabel(r'$A2_{t}$', ha='center', labelpad=20, rotation=0, fontsize=18)

axS1.set_ylabel(r'$S1$', ha='center', labelpad=20, rotation=0, fontsize=18)
axS2.set_ylabel(r'$S2$', ha='center', labelpad=20, rotation=0, fontsize=18)



axes = [ ax1, axA1, axA2, ax1T, axA1T, axA2T, axS1, axS2]
for ax in axes:
    ax.grid()
    if args.xkey != aSkey:
        ax.set_xlabel( args.xkey ) 
    else:
        ax.set_xlabel('$U/t$')

    #for l in ax.xaxis.get_ticklabels():
    #    l.set_rotation(30)
    #ax.xaxis.set_major_formatter(\
    #     matplotlib.ticker.FormatStrFormatter( '%d$^{\circ}$' ) )
    #ax1.xaxis.set_major_locator( matplotlib.ticker.MultipleLocator(0.1) )

    ax.legend(loc='best',numpoints=1,prop={'size':7},\
               handlelength=1.1,handletextpad=0.5)

axA1T.set_ylim( axS1.get_ylim() ) 
axA2T.set_ylim( axS2.get_ylim() ) 

#axA2T.set_ylim(0.8,1.6)
#axA1T.set_ylim(0.7,1.1)
#ax1T.set_ylim(0.8,1.8)



for pos in ['top','bottom','right','left']:
    for ax in [axA2T, axS2]:
        ax.spines[pos].set_edgecolor('green')
        ax.spines[pos].set_linewidth(2.0) 
    for ax in [axA1T, axS1]:
        ax.spines[pos].set_edgecolor('purple')
        ax.spines[pos].set_linewidth(2.0) 
 
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

