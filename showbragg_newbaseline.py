#!/usr/bin/python

import sys
import argparse
import os
import glob
from configobj import ConfigObj

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import itertools

sys.path.append('/lab/software/apparatus3/py')

import falsecolor
import gaussfitter
import qrange
import statdat
import scipy

from uncertainties import ufloat


# --------------------- MAIN CODE  --------------------#

if __name__ == "__main__":
  parser = argparse.ArgumentParser('qstat.py')

  parser.add_argument('BGKEY', action="store", \
         type=str, help='using this key as discriminator for the background shots')

  parser.add_argument('XKEY', action="store",\
         help='use this key to produce bragg plot')

  parser.add_argument('YKEYS', action="store", nargs='*',\
         help='get statistics for all keys in this list')

  parser.add_argument('--range', action = "store", \
         help="range of shots to be used.")

  parser.add_argument('--output', action="store", \
         help="optional path of png to save figure") 


  args = parser.parse_args()

  print args
 

  rangestr = args.range.replace(':','to')
  rangestr = rangestr.replace(',','_')
  shots = qrange.parse_range(args.range)

  if 'SEQ:shot' not in args.YKEYS:
      args.YKEYS.append('SEQ:shot')

  datadict = {} 
  for s in shots:
      report = ConfigObj( 'report'+s+'.INI')  
      xkey = qrange.evalstr( report, args.XKEY  )
      if xkey not in datadict.keys():
          datadict[xkey] = {}
 
      bgkey = qrange.evalstr( report, args.BGKEY ) 
      if bgkey not in datadict[xkey].keys():
          datadict[xkey][bgkey] = {} 
     
      for Y in args.YKEYS:
          if Y not in datadict[xkey][bgkey].keys(): 
              datadict[xkey][bgkey][Y] = []
          val = qrange.evalstr(report, Y)
          datadict[xkey][bgkey][Y].append( val  )
          if val is np.nan:
              print "Encountered nan in Shot #", s
  import pprint
  #pprint.pprint( datadict )

  #This defines the value of the BGKEY that is used for background
  bgdval = 0.0 
  #This defines the value of the BGKEY that is used for signal 
  sigval = 1.0
  #Any other bgvals are ignored

  #This defines the Bragg figure of merit to be used
  bragg = 'HHHEIGEN:andor2norm'


  ##### START PRODUCING THE PLOT #####
  
  Yiter = args.YKEYS
  Yiter.remove('SEQ:shot')
  mainplotrows = max( len( datadict.keys()) / 2 , 1)
  rows = mainplotrows + len( Yiter ) 
  cols = len( datadict.keys()) 
  fig = plt.figure( figsize=( 3*cols, 2*rows) )

  y0 = { 'ANDOR1EIGEN:signal' : 0. , 
         'ANDOR2EIGEN:signal' : 0. ,
         'HHHEIGEN:andor2norm' : 0. } 
  y1 = { 'ANDOR1EIGEN:signal' : 60000. , 
         'ANDOR2EIGEN:signal' : 18000. ,
         'HHHEIGEN:andor2norm' : 0.80 } 
  
  data = [] 
  bdata = []
  allbdata = [] 

 
  for i,xk in enumerate(sorted(datadict.keys())):
      sigdata = datadict[xk][sigval][ bragg ]
      sigdata = np.ma.masked_invalid( sigdata)
      sigdata = sigdata.compressed()
      data.append( [ xk, np.mean(sigdata), scipy.stats.sem(sigdata) ] ) 

      try:
        bgdata = datadict[xk][bgdval][ bragg ]
        bgdata =  np.ma.masked_invalid( bgdata)
        bgdata = bgdata.compressed() 
        bdata.append( [ xk, np.mean(bgdata), scipy.stats.sem(bgdata) ] ) 
      except:
        pass
 
      try:
        allbdata = allbdata + datadict[xk][bgdval][ bragg ] 
      except:
        pass

      for j,Y in enumerate(Yiter):

          
          axbg = plt.subplot2grid( (rows, cols), (j,i), rowspan=1, colspan=1) 
          bbox = axbg.get_position().get_points()
          xw = bbox[1,0] - bbox[0,0]
          yw = bbox[1,1] - bbox[0,1] 
          newbbox = matplotlib.transforms.Bbox.from_bounds( bbox[0,0], bbox[0,1], xw/2., yw)
          axbg.set_position(newbbox)
          try:
            axbg.plot( datadict[xk][bgdval]['SEQ:shot'], datadict[xk][bgdval][ Y ], 'o') 
            bgmean = np.mean(  datadict[xk][bgdval][ Y ] )
            bgstderr = scipy.stats.sem( datadict[xk][bgdval] [ Y ] )  
            axbg.axhspan( bgmean-bgstderr, bgmean+bgstderr, facecolor='blue', alpha=0.6, linewidth=0)
            axbg.text(0.05, 0.02, "%.3f\n+/- %.3f" % (bgmean, bgstderr), \
                      fontsize=10, weight='bold',\
                      ha='left', va='bottom', transform = axbg.transAxes)
          except:
            pass

          #plt.delaxes(axbg)
          #axbg = fig.add_axes( [bbox[0,0], bbox[0,1], xw/2., yw] ) 
          axsg = fig.add_axes( [bbox[0,0]+xw/2., bbox[0,1], xw/2, yw] )
          axsg.plot( datadict[xk][sigval]['SEQ:shot'], datadict[xk][sigval][ Y ], 'o' )
          sgmean = np.mean(  datadict[xk][sigval][ Y ] )
          sgstderr = scipy.stats.sem( datadict[xk][sigval] [ Y ] )  
          axsg.axhspan( sgmean-sgstderr, sgmean+sgstderr, facecolor='blue', alpha=0.6, linewidth=0)
          axsg.text(0.05, 0.02, "%.3f\n+/- %.3f" % (sgmean, sgstderr), \
                    fontsize=10, weight='bold',\
                    ha='left', va='bottom', transform = axsg.transAxes)

          #axbg.set_ylim( y0[Y], y1[Y] ) 
          #axsg.set_ylim( y0[Y], y1[Y] )
          sigvals = datadict[xk][sigval][ Y ] 
          bgdvals = datadict[xk][bgdval][ Y ] 
          maxY = max( sigvals + bgdvals )  
          minY = min( sigvals + bgdvals )
          padding = np.abs(maxY - minY) * 0.2
          axbg.set_ylim( minY - padding, maxY + padding )
          axsg.set_ylim( minY - padding, maxY + padding ) 
          #axbg.set_ylim( 0., maxY + padding )
          #axsg.set_ylim( 0., maxY + padding ) 
  

          for tick in axsg.xaxis.get_major_ticks():
                tick.label.set_fontsize(7) 
                tick.label.set_rotation(70)
          for tick in axbg.xaxis.get_major_ticks():
                tick.label.set_fontsize(7) 
                tick.label.set_rotation(70)

          for tick in axsg.yaxis.get_major_ticks():
                tick.label.set_fontsize(8)
          for tick in axbg.yaxis.get_major_ticks():
                tick.label.set_fontsize(8) 

          axbg.grid()
          axsg.grid()
          axsg.yaxis.set_ticklabels([])

          if i == 0:
              axbg.set_ylabel(Y)
    
          if j == 0:
              axbg.set_title('BGND')
              axsg.set_title('SIGNAL')
 


           
  scale = cols/2. 
  msscale = cols/1.
                      
  data = np.array(data)
  bdata = np.array(bdata)


  print "rows = ", rows
  print "cols = ", cols
  print "mainplotrows = ", mainplotrows
  ax = plt.subplot2grid( (rows,cols), (len(Yiter), 0), rowspan=mainplotrows, colspan=cols)

  ax.errorbar( data[:,0], data[:,1] , yerr=data[:,2], \
               capsize=0., elinewidth = 1.0*msscale ,\
               fmt='.', ecolor='blue', mec='blue', \
               mew=1.0*msscale, ms=4.0*msscale,\
               marker='o', mfc='lightblue', \
               label="signal")  

  ax.errorbar( bdata[:,0], bdata[:,1] , yerr=bdata[:,2], \
               capsize=0., elinewidth = 1.0*msscale ,\
               fmt='.', ecolor='black', mec='black', \
               mew=1.0*msscale, ms=4.0*msscale,\
               marker='o', mfc='gray',\
               label="baseline")

  ax.legend(loc='best', numpoints=1,prop={'size':10*scale} ) 
  
  allbdata = np.array(allbdata)
  allbdata = np.ma.masked_invalid( allbdata)
  allbdata = allbdata.compressed() 
  zero = np.mean( allbdata)
  zero_stderr = scipy.stats.sem( allbdata )
  
  ax.axhspan( zero-zero_stderr, zero+zero_stderr, facecolor='gray', alpha=0.6, linewidth=0)
  
  #ax.axvspan( 29.7, 33.7, facecolor='red', alpha=0.6, linewidth=0)
  #ax.axvspan( 105.7, 109.7, facecolor='red', alpha=0.6, linewidth=0)

  maxX = bdata[:,0].max()   
  minX = bdata[:,0].min()
  xpadding = np.abs( maxX - minX) * 0.15
  if xpadding ==  0.:
    xpadding = (ax.get_xlim()[1] - ax.get_xlim()[0])*0.15
  else:
    ax.set_xlim( minX - xpadding, maxX + xpadding )
  ax.set_xlabel( args.XKEY, fontsize = 12*scale )
  ax.set_ylabel( 'Normalized Bragg', fontsize = 12*scale )

  for i,d in enumerate(data):
    sigi = data[i][1]
    sigierr = data[i][2] 
    bgdi = bdata[i][1]
    bgdierr = data[i][2] 

    s = ufloat( ( sigi, sigierr ) ) 
    b = ufloat( ( bgdi, bgdierr ) ) 

    ratio =  s / b 

    ax.text( d[0]+xpadding/5.*msscale, d[1], '%.2f\n' % ratio.nominal_value +'$\pm$%.2f' % ratio.std_dev(), \
             fontsize=7.*msscale , va='center') 

  for tick in ax.xaxis.get_major_ticks():
      tick.label.set_fontsize(12*scale)
  for tick in ax.yaxis.get_major_ticks():
      tick.label.set_fontsize(12*scale)
  
  ax.grid()
 
  if args.output != None:
    print "Saving figure to %s" % args.output
    fig.savefig( args.output, dpi=120 ) 
  else:
    plt.show() 
  
   

 

  

