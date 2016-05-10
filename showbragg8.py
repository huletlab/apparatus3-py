#!/usr/bin/python

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
sys.path.append('/lab/software/apparatus3/py')
import qrange, statdat, fitlibrary
from uncertainties import ufloat,unumpy
import scipy

import argparse
parser = argparse.ArgumentParser('showbragg')

parser.add_argument('BGKEY', action="store", \
       type=str, help='using this key as discriminator for the background shots')

parser.add_argument('XKEY', action="store",\
       help='use this key to produce bragg plot')

parser.add_argument('--range', action = "store", \
       help="range of shots to be used.")

parser.add_argument('--tofval', action = "store", type=float,\
       help="value of tof to be used for bgnd.")

parser.add_argument('--tofimgdet', default=-135.0, action = "store", type=float,\
       help="value of tof to be used for bgnd.")

parser.add_argument('--resimgdet', default=-172.4, action = "store", type=float,\
       help="value of tof to be used for on res shots.")

parser.add_argument('--output', action="store", \
       help="optional path of png to save figure") 

parser.add_argument('--ratio', action="store_true", \
       help="optional to make plot of signal/base ratio") 

parser.add_argument('--singlecams', action="store_true", \
       help="optional to make plot of single camera ratios") 

parser.add_argument('--singleres', action="store_true", \
       help="optional to make plot of single camera ratios") 

parser.add_argument('--reanalyzed', action="store_true", \
       help="optional to make plot of signal/base ratio") 

parser.add_argument('--concise', action="store_true", \
       help="optional to output only the final results") 

args = parser.parse_args()

if args.tofval is None:
    args.tofval = 0.25

#print args
verbose = not args.concise
CONCISEOUT = '' 

rangestr = args.range.replace(':','to')
rangestr = rangestr.replace(',','_')
shots = qrange.parse_range(args.range)

if args.reanalyzed == True:
    SIGNALKEY = 'HHHEIGEN_re:andor2norm'
    ANDOR1KEY = 'ANDOR1EIGEN_re:signal'
    ANDOR2KEY = 'ANDOR2EIGEN_re:signal'
else:
    SIGNALKEY = 'HHHEIGEN:andor2norm'
    ANDOR1KEY = 'ANDOR1EIGEN:signal'
    ANDOR2KEY = 'ANDOR2EIGEN:signal'


# Setup data locations and fetch data
cdir = os.getcwd()
BraggKeys = [SIGNALKEY, ANDOR1KEY, ANDOR2KEY, 'DIMPLELATTICE:imgdet', 'SEQ:shot', 'DIMPLELATTICE:force_lcr3']
AllKeys = BraggKeys + [args.XKEY, args.BGKEY]
print "\nObtaining data from : ", AllKeys
def K( key ):
    try:
        index = AllKeys.index(key)
        return index
    except Exception as e:
        print e 
        exit()
data, errmsg, rawdat = qrange.qrange_eval( cdir, args.range, AllKeys)

# Deal with the backgrounds
if args.BGKEY == 'DIMPLELATTICE:force_lcr3':
  #This defines the value of the BGKEY that is used for background
  bgdval = 0.0 
  #This defines the value of the BGKEY that is used for signal 
  sigval = 0.0 
elif args.BGKEY == 'DIMPLELATTICE:signal':
  bgdval = 0.
  sigval = 1.
elif args.BGKEY == 'DIMPLELATTICE:tof':
  bgdval = args.tofval
  sigval = 0.
  # Remove everything that has DL:force_lcr3 == 0
  cond1 = data[:,K('DIMPLELATTICE:force_lcr3')] != 0
  cond2 = data[:,K('DIMPLELATTICE:imgdet')] == args.tofimgdet 
  cond3 = data[:,K('DIMPLELATTICE:tof')] == 0
  mask = np.logical_and(cond1,cond2)
  mask = np.logical_or(mask,cond3)
  data = data[ mask ]

#Any other bgvals are ignored
bgdat = data[ data[:, K(args.BGKEY)] == bgdval] 
sgdat = data[ data[:, K(args.BGKEY)] == sigval]
remain = data[ np.logical_and( data[:,K(args.BGKEY)] != bgdval ,data[:,K(args.BGKEY)] != sigval )]
if remain.size > 0:
    "Background key has more than two possible values in this set."


# Find all possible values of the XKEY
xvals = np.unique( data[:, K(args.XKEY)] ) 

# Find all possible values of the XKEY that are NOT resonant shots
# These are the subplots that are going to go on the top

def resonant( FLAG, datarray):
    boolarray = np.abs( datarray[:,K('DIMPLELATTICE:imgdet')] - (args.resimgdet)) > 5 
    if FLAG == False:
        return datarray[ boolarray ]
    else:
        return datarray[ np.logical_not( boolarray ) ] 

dataNoRes = resonant( False, data) 
xvalsP = np.unique( dataNoRes[:, K(args.XKEY)] )
print "\nXVALS = ",xvalsP
sigNoRes  = resonant( False, sgdat)
sigRes    = resonant( True,  sgdat )

# Start making the plot
Yplots = 3
mainplotrows = max( (1+Yplots )/2 + xvalsP.size/4 , 1)
rows = mainplotrows + Yplots 
#
if args.singlecams == True:
    rows+= mainplotrows
    args.ratio = True
#
cols = xvalsP.size
if args.ratio == True and cols ==1:
    cols +=1

gs = matplotlib.gridspec.GridSpec( rows,cols)
if cols < 2:
  leftspace = 0.25 
elif cols < 4:
  leftspace = 0.15
elif cols < 6: 
  leftspace = 0.10
else:
  leftspace = 0.05
gs.update( left=leftspace, hspace=0.45, wspace=0.4 ) 
fig = plt.figure( figsize=( 4*cols, 2*rows) )


####
#### START OUT BY MAKING THE SMALL PLOTS AT THE TOP
####
ylims={}
for i,xk in enumerate(xvalsP): 
    for j,Y in enumerate(range(Yplots)):

        axbg = plt.subplot( gs[j,i] ) 
        bbox = axbg.get_position().get_points()
        xw = bbox[1,0] - bbox[0,0]
        yw = bbox[1,1] - bbox[0,1] 
        newbbox = matplotlib.transforms.Bbox.from_bounds( bbox[0,0], bbox[0,1], xw/3., yw)
        axbg.set_position(newbbox)

        try:
            bxk = bgdat[ bgdat[:,K(args.XKEY)] == xk ]
            axbg.plot( bxk[:,K('SEQ:shot')], bxk[:,j], 'o', color='black') 
            bgmean = np.mean(  bxk[:,j]  )
            bgstderr = scipy.stats.sem( bxk[:,j]  )  
            axbg.axhspan( bgmean-bgstderr, bgmean+bgstderr, facecolor='black', alpha=0.6, linewidth=0)
            axbg.text(0.05, 0.02, "%.3f\n+/- %.3f" % (bgmean, bgstderr), \
                      fontsize=9, weight='bold',\
                      ha='left', va='bottom', transform = axbg.transAxes)
        except:
            bxk = np.zeros(shape=(1,3)) 
        

        axsg = fig.add_axes( [bbox[0,0]+xw/3., bbox[0,1], xw/3., yw] )
        sxk = sigNoRes[ sigNoRes[:,K(args.XKEY)] == xk ]
        axsg.plot( sxk[:,K('SEQ:shot')], sxk[:,j], 'o', color='blue') 
        sgmean = np.mean( sxk[:,j]  )
        sgstderr = scipy.stats.sem( sxk[:,j]  )  
        axsg.axhspan( sgmean-sgstderr, sgmean+sgstderr, facecolor='blue', alpha=0.6, linewidth=0)
        axsg.text(0.05, 0.02, "%.3f\n+/- %.3f" % (sgmean, sgstderr), \
                  fontsize=9, weight='bold',\
                  ha='left', va='bottom', transform = axsg.transAxes)

        axres = fig.add_axes( [bbox[0,0]+xw*2./3., bbox[0,1], xw/3., yw] )
        try:
          rxk = sigRes[ sigRes[:,K(args.XKEY)] == xk ]
          axres.plot( rxk[:,K('SEQ:shot')], rxk[:,j], 'o', color='red')
          resmean = np.mean( rxk[:,j] )
          resstderr = scipy.stats.sem( rxk[:,j])
          axres.axhspan( resmean-resstderr, resmean+resstderr, facecolor='red', alpha=0.6, linewidth=0)
          axres.text(0.05, 0.02, "%.3f\n+/- %.3f" % (resmean, resstderr), \
                  fontsize=9, weight='bold',\
                  ha='left', va='bottom', transform = axres.transAxes)

          maxY = rxk[:,j].max()   
          minY = rxk[:,j].min() 
          padding = np.abs(maxY - minY) * 0.25
          ylim0 = minY-3.0*padding 
          ylim1 = maxY + padding
          if ylim0 == ylim1:
              ylim0 = 0.8*ylim0
              ylim1 = 1.2*ylim1
          axres.set_ylim( ylim0, ylim1)
        except:
          rxk = np.zeros_like(sxk)
        axres.yaxis.tick_right()
        for tick in axres.yaxis.get_major_ticks():
          tick.label2.set_fontsize(7.)
        axres.yaxis.set_label_position('right')
         

        maxY = max( sxk[:,j].max() , bxk[:,j].max() )  
        minY = min( sxk[:,j].min() , bxk[:,j].min() )
        padding = np.abs(maxY - minY) * 0.25
        ylim0 = minY-3.0*padding 
        ylim1 = maxY + padding
        axbg.set_ylim( ylim0, ylim1)
        axsg.set_ylim( ylim0, ylim1)

        Y = BraggKeys[j]
        if Y in ylims.keys():
            ylims[Y].append(axbg) 
            ylims[Y].append(axsg)
        else:
            ylims[Y]= [ axbg, axsg ] 
        #axbg.set_ylim( 0., maxY + padding )
        #axsg.set_ylim( 0., maxY + padding ) 

        allaxes = [axsg, axbg, axres]
        for ax in allaxes:
            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(7) 
                tick.label.set_rotation(70)
            if ax is not axres:
              for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(8)
            ax.grid()
            
      
        axbg.xaxis.get_major_ticks()[-1].set_visible(False)
        axsg.xaxis.get_major_ticks()[0].set_visible(False)
        axres.xaxis.get_major_ticks()[0].set_visible(False)

        axsg.yaxis.set_ticklabels([])

        if i == 0:
            if 'ANDOR1' in Y:
              axbg.set_ylabel('Andor1')
            elif 'ANDOR2' in Y:
              axbg.set_ylabel('Andor2')
            elif 'andor2norm' in Y:
              axbg.set_ylabel('A2/A1')
            else:
              axbg.set_ylabel(Y)
  
        if j == 0:
            axbg.set_title('TOF')
            axsg.set_title('IN-SITU')
            axres.set_title('RES')

for yk in ylims.keys():
  mins = [ a.get_ylim()[0] for a in ylims[yk] ] 
  maxs = [ a.get_ylim()[1] for a in ylims[yk] ] 
  for a in ylims[yk]:
    if max(maxs) > 10.:
      a.set_ylim( -2000., max(maxs) ) 
    else:
      a.set_ylim( min(mins), max(maxs) ) 

####
#### THEN MAKE THE BIG PLOT AT THE BOTTOM 
####
scale = (cols**0.5)/3.0
msscale = (cols**0.5)/3.5
if verbose:
  print "rows = ", rows
  print "cols = ", cols
  print "mainplotrows = ", mainplotrows
if args.ratio == True:
  ax = plt.subplot( gs[ Yplots:Yplots+mainplotrows, : (cols+1)/2 ] )
else:
  ax = plt.subplot( gs[ Yplots:Yplots+mainplotrows, :  ] )
if args.ratio == True:
  ax2 = plt.subplot( gs[ Yplots:Yplots+mainplotrows, (cols+1)/2 : ] )

xdat       = statdat.statdat(sgdat,    K(args.XKEY), K(SIGNALKEY))
xdatNoRes  = statdat.statdat(sigNoRes, K(args.XKEY), K(SIGNALKEY))
xdatRes    = statdat.statdat(sigRes,   K(args.XKEY), K(SIGNALKEY))
bxdat      = statdat.statdat(bgdat,    K(args.XKEY), K(SIGNALKEY))


# Signal points
ax.errorbar( xdatNoRes[:,0], xdatNoRes[:,1] , yerr=xdatNoRes[:,3], \
             capsize=0., elinewidth = 2.75*msscale ,\
             fmt='.', ecolor='blue', mec='blue', \
             mew=2.75*msscale, ms=11.0*msscale,\
             marker='o', mfc='lightblue', \
             label="signal") 

# Background points (these are typically TOF)
try:
    ax.errorbar( bxdat[:,0], bxdat[:,1] , yerr=bxdat[:,3], \
                 capsize=0., elinewidth = 2.75*msscale ,\
                 fmt='.', ecolor='black', mec='black', \
                 mew=2.75*msscale, ms=11.0*msscale,\
                 marker='o', mfc='gray',\
                 label="baseline")
    zero = np.mean( bgdat[:,K(SIGNALKEY)] )
    zero_stderr = scipy.stats.sem( bgdat[:,K(SIGNALKEY)] )
    ax.axhspan( zero-zero_stderr, zero+zero_stderr, facecolor='gray', alpha=0.6, linewidth=0)
except:
    pass

# Points on Resonance 
try: 
  ax.errorbar( xdatRes[:,0], xdatRes[:,1] , yerr=xdatRes[:,3], \
             capsize=0., elinewidth = 2.75*msscale ,\
             fmt='.', ecolor='red', mec='red', \
             mew=2.75*msscale, ms=11.0*msscale,\
             marker='o', mfc='pink', \
             label="signal")  
except:
  pass

# Put labels on the points
#for i,d in enumerate(xdat):
#    print d
#    continue

# Adjust X limits for the A2/A1 plot and set labels
try: 
    maxX = np.amax( np.concatenate(( bxdat[:,0], xdat[:,0])))
    minX = np.amin( np.concatenate(( bxdat[:,0], xdat[:,0])))
    xpadding = np.abs( maxX - minX) * 0.2 / cols
    if xpadding ==  0.:
      if verbose:
        print 'Only one X point in set'
      xpadding = (ax.get_xlim()[1] - ax.get_xlim()[0])*0.05
    ax.set_xlim( minX - xpadding, maxX + 2.*xpadding )
except:
    maxX = None
    minX = None 
ax.set_xlabel( args.XKEY, fontsize = 22.*scale )
ax.set_ylabel( 'A2/A1', fontsize = 22.*scale )

# Adjust the tick label sizes
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(20.*scale)
    #  tick.label.set_rotation(15)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(20.*scale)
ax.grid()


####
#### MAKE THE TOF and RES BIG PLOT AT THE BOTTOM 
####
if args.ratio == True:
    if xdatNoRes.shape != bxdat.shape or xdatRes.shape != bxdat.shape:
        print "Can't calculate ratio, Different number of signal and bgnd points"
    else:
        xdatResU = unumpy.uarray(( xdatRes[:,1].tolist(), xdatRes[:,3].tolist() ))
        xdatU    = unumpy.uarray(( xdatNoRes[:,1].tolist(), xdatNoRes[:,3].tolist() ))
        bxdatU   = unumpy.uarray(( bxdat[:,1].tolist(), bxdat[:,3].tolist() ))
        #
        sxdatU = xdatU[ np.argsort( xdatNoRes[:,0] )]
        bxdatU = bxdatU[ np.argsort( bxdat[:,0] )]
        ratioTOF = sxdatU / bxdatU
        #print "DEBUGGING RATIO"
        #print ratioTOF 
        ax2.errorbar( xdatNoRes[:,0][np.argsort(xdatNoRes[:,0])], unumpy.nominal_values(ratioTOF),\
                      yerr = unumpy.std_devs(ratioTOF), \
                   capsize=0., elinewidth = 2.75*msscale ,\
                   fmt='.', ecolor='green', mec='green', \
                   mew=2.75*msscale, ms=11.0*msscale,\
                   marker='o', mfc='limegreen', \
                   label="signal/base")
         
        xdatResU = xdatResU[ np.argsort( xdatRes[:,0] )] 
        xdatURes = xdatU[ np.argsort( xdatRes[:,0] )]
        ratioRES = xdatURes/xdatResU 
        ax2.errorbar( xdatRes[:,0][np.argsort(xdatRes[:,0])], unumpy.nominal_values( ratioRES  ),\
                      yerr = unumpy.std_devs( ratioRES ), \
                   capsize=0., elinewidth = 2.75*msscale ,\
                   fmt='.', ecolor='red', mec='red', \
                   mew=2.75*msscale, ms=11.0*msscale,\
                   marker='o', mfc='pink', \
                   label="signal/base")
       
        maxY = unumpy.nominal_values(ratioTOF).max()
        #ax2.set_ylim( 0.9, maxY+0.05 )
        ax2.set_ylim( 0.5, 3.0 )
        padding = (ax2.get_xlim()[1] - ax2.get_xlim()[0])*0.1
        ax2.set_xlim( ax2.get_xlim()[0] - padding ,ax2.get_xlim()[1] + padding)
        ax2.set_ylabel('A2/A1 / A2/A1 (TOF)')
  
        ax2.axhspan( 0.99, 1.01, facecolor='green', alpha=0.6, linewidth=0)
        for tick in ax2.xaxis.get_major_ticks():
          tick.label.set_fontsize(20.*scale)
        ax2.yaxis.tick_right()
        for tick in ax2.yaxis.get_major_ticks():
          tick.label2.set_fontsize(20.*scale)
        ax2.yaxis.set_label_position('right')
      
        ax2.grid()


####
#### MAKE THE A1, A2 PLOTS AT THE BOTTOM 
####
if args.singlecams == True:
  ax3 = plt.subplot( gs[ Yplots+mainplotrows:, : (cols+1)/2 ] )
  ax4 = plt.subplot( gs[ Yplots+mainplotrows:, (cols+1)/2 : ] )

  def plotCam( KEY, axplot ):
      A1dat      = statdat.statdat(sigNoRes, K(args.XKEY), K(KEY))
      A1res      = statdat.statdat(sigRes,   K(args.XKEY), K(KEY))
      A1tof      = statdat.statdat(bgdat,    K(args.XKEY), K(KEY))
    
      A1datU = unumpy.uarray(( A1dat[:,1].tolist(), A1dat[:,3].tolist() ))
      try: 
          A1resU = unumpy.uarray(( A1res[:,1].tolist(), A1res[:,3].tolist() )) 
          plotres = True
      except:
          A1resU = unumpy.uarray(( [1.], [1.] ))
          plotres = False

      try:
          A1tofU = unumpy.uarray(( A1tof[:,1].tolist(), A1tof[:,3].tolist() ))
          plottof = True
      except:
          A1tofU = np.ones_like(A1datU)
          plottof = False
          
        

      A1datU = A1datU[ np.argsort( A1dat[:,0] )]  
      try:
          A1resU = A1resU[ np.argsort( A1res[:,0] )] 
          plotres = plotres and True
      except:
          A1resU = A1resU
          plotres = False
    
      if plottof == True:
          A1tofU = A1tofU[ np.argsort( A1tof[:,0] )] 

   
      xvals =  A1dat[:,0][np.argsort(A1dat[:,0])]
      yunc  =  A1datU/A1tofU
      yvals =  unumpy.nominal_values(yunc)
      yerrvals  =  unumpy.std_devs(yunc)
      axplot.errorbar( xvals, yvals, yerr=yerrvals,\
                 capsize=0., elinewidth = 2.75*msscale ,\
                 fmt='.', ecolor='green', mec='green', \
                 mew=2.75*msscale, ms=11.0*msscale,\
                 marker='o', mfc='limegreen', \
                 label="signal/base")
   
      if plotres and args.singleres: 
          tw = axplot.twinx()
          tw.errorbar( A1dat[:,0][np.argsort(A1dat[:,0])], unumpy.nominal_values(A1datU/A1resU),\
                        yerr = unumpy.std_devs(A1datU/A1resU), \
                     capsize=0., elinewidth = 2.75*msscale ,\
                     fmt='.', ecolor='red', mec='red', \
                     mew=2.75*msscale, ms=11.0*msscale,\
                     marker='o', mfc='pink', \
                     label="signal/base")
      axplot.grid()
      axplot.set_xlabel(args.XKEY)
      if 'ANDOR1' in KEY:
          axplot.set_ylabel("A1/A1_tof")
          print "A1/A1tof:\n\t", print_uarray(yunc)
      elif 'ANDOR2' in KEY:
          axplot.set_ylabel("A2/A2_tof")
          print "A2/A2tof:\n\t", print_uarray(yunc)
      else:
          axplot.set_ylabel(KEY)
  
  def print_uarray( ua, fmt="%.3f"):
      val = unumpy.nominal_values(ua)
      err = unumpy.std_devs(ua) 
      us = []
      for i in range(len(val)):
          us.append( '+/-'.join([fmt,fmt])%(val[i],err[i]) )
          
      fmtstr = ', '.join(us)
      return fmtstr
      

  print 
  plotCam( ANDOR1KEY, ax3)
  plotCam( ANDOR2KEY, ax4) 
  print 
  



if args.output != None:
  if verbose:
    print "Saving figure to %s" % args.output
  fig.savefig( args.output, dpi=120 ) 
else:
  plt.show() 

exit()
 


