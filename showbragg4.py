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

import datetime


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

  parser.add_argument('--xlabels', action="store", nargs='*', \
         help="optional path of png to save figure")

  parser.add_argument('--ratio', action="store_true", \
         help="optional to make plot of signal/base ratio") 

  parser.add_argument('--anglex', action="store_true", \
         help="optional to convert the x axis using the Bragg angle table") 

  parser.add_argument('--concise', action="store_true", \
         help="optional to output only the final results") 

  args = parser.parse_args()

  #print args

  verbose = not args.concise
  CONCISEOUT = '' 
 

  rangestr = args.range.replace(':','to')
  rangestr = rangestr.replace(',','_')
  shots = qrange.parse_range(args.range)

  if 'SEQ:shot' not in args.YKEYS:
      args.YKEYS.append('SEQ:shot')

  datadict = {} 
  timekey = 'SEQ:abstime'
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
              if verbose:
                print "Encountered nan in Shot #", s

      #Also include timestamp information
      datadict[xkey][bgkey][timekey] = qrange.evalstr(report,timekey)
 
  import pprint
  #pprint.pprint( datadict )

  if args.BGKEY == 'DIMPLELATTICE:force_lcr3':
    #This defines the value of the BGKEY that is used for background
    bgdval = 0.0 
    #This defines the value of the BGKEY that is used for signal 
    sigval = -1.0 
  elif args.BGKEY == 'DIMPLELATTICE:signal':
    bgdval = 0.
    sigval = 1.
  elif args.BGKEY == 'DIMPLELATTICE:tof':
    bgdval = 0.5
    sigval = 0.
  #Any other bgvals are ignored

  #This defines the Bragg figure of merit to be used
  bragg = 'HHHEIGEN:andor2norm'


  ##### START PRODUCING THE PLOT #####
  
  Yiter = args.YKEYS
  Yiter.remove('SEQ:shot')
  if verbose:
    print datadict.keys()
  mainplotrows = max((1+ len( datadict.keys())) / 2 , 1)
  rows = mainplotrows + len( Yiter ) 
  cols = len( datadict.keys())
  gs = matplotlib.gridspec.GridSpec( rows,cols)
  if cols < 2:
    leftspace = 0.25 
  elif cols < 4:
    leftspace = 0.15
  elif cols < 6: 
    leftspace = 0.10
  else:
    leftspace = 0.05
  gs.update( left=leftspace, hspace=0.3) 
 
  fig = plt.figure( figsize=( 3*cols, 2*rows) )

  y0 = { 'ANDOR1EIGEN:signal' : 0. , 
         'ANDOR2EIGEN:signal' : 0. ,
         'HHHEIGEN:andor2norm' : 0. } 
  y1 = { 'ANDOR1EIGEN:signal' : 60000. , 
         'ANDOR2EIGEN:signal' : 18000. ,
         'HHHEIGEN:andor2norm' : 0.80 } 
  
  data = [] 
  bdata = []
  timestamp = [] 
  allbdata = [] 

  ylims={}
 
  for i,xk in enumerate(sorted(datadict.keys())):

      timestamp.append( np.mean( datadict[xk][sigval][timekey] ) )

      sigdata = datadict[xk][sigval][ bragg ]
      sigdata = np.ma.masked_invalid( sigdata)
      sigdata = sigdata.compressed()

      A1dat  = datadict[xk][sigval][ 'ANDOR1EIGEN:signal' ] 
      A2dat  = datadict[xk][sigval][ 'ANDOR2EIGEN:signal' ] 

      data.append( [ xk, np.mean(sigdata), scipy.stats.sem(sigdata), \
                         np.mean(A1dat), scipy.stats.sem(A1dat), \
                         np.mean(A2dat), scipy.stats.sem(A2dat) ] )

      try:
        bgdata = datadict[xk][bgdval][ bragg ]
        bgdata =  np.ma.masked_invalid( bgdata)

        A1dat  = datadict[xk][bgdval][ 'ANDOR1EIGEN:signal' ] 
        A2dat  = datadict[xk][bgdval][ 'ANDOR2EIGEN:signal' ] 

        bgdata = bgdata.compressed() 
        bdata.append( [ xk, np.mean(bgdata), scipy.stats.sem(bgdata), \
                            np.mean(A1dat), scipy.stats.sem(A1dat), \
                            np.mean(A2dat), scipy.stats.sem(A2dat) ] )
      except:
        pass
 
      try:
        allbdata = allbdata + datadict[xk][bgdval][ bragg ] 
      except:
        pass

      for j,Y in enumerate(Yiter):

          axbg = plt.subplot( gs[j,i] ) 
          #axbg = plt.subplot2grid( (rows, cols), (j,i), rowspan=1, colspan=1) 
          bbox = axbg.get_position().get_points()
          xw = bbox[1,0] - bbox[0,0]
          yw = bbox[1,1] - bbox[0,1] 
          newbbox = matplotlib.transforms.Bbox.from_bounds( bbox[0,0], bbox[0,1], xw/2., yw)
          axbg.set_position(newbbox)
          try:
            axbg.plot( datadict[xk][bgdval]['SEQ:shot'], datadict[xk][bgdval][ Y ], 'o',color='black') 
            bgmean = np.mean(  datadict[xk][bgdval][ Y ] )
            bgstderr = scipy.stats.sem( datadict[xk][bgdval] [ Y ] )  
            axbg.axhspan( bgmean-bgstderr, bgmean+bgstderr, facecolor='black', alpha=0.6, linewidth=0)
            axbg.text(0.05, 0.02, "%.3f\n+/- %.3f" % (bgmean, bgstderr), \
                      fontsize=10, weight='bold',\
                      ha='left', va='bottom', transform = axbg.transAxes)
          except:
            pass

          #plt.delaxes(axbg)
          #axbg = fig.add_axes( [bbox[0,0], bbox[0,1], xw/2., yw] ) 
          axsg = fig.add_axes( [bbox[0,0]+xw/2., bbox[0,1], xw/2, yw] )
          axsg.plot( datadict[xk][sigval]['SEQ:shot'], datadict[xk][sigval][ Y ], 'o', color='blue' )
          sgmean = np.mean(  datadict[xk][sigval][ Y ] )
          sgstderr = scipy.stats.sem( datadict[xk][sigval] [ Y ] )  
          axsg.axhspan( sgmean-sgstderr, sgmean+sgstderr, facecolor='blue', alpha=0.6, linewidth=0)
          axsg.text(0.05, 0.02, "%.3f\n+/- %.3f" % (sgmean, sgstderr), \
                    fontsize=10, weight='bold',\
                    ha='left', va='bottom', transform = axsg.transAxes)

          #axbg.set_ylim( y0[Y], y1[Y] ) 
          #axsg.set_ylim( y0[Y], y1[Y] )
          sigvals = datadict[xk][sigval][ Y ]
          try: 
            bgdvals = datadict[xk][bgdval][ Y ]
          except:
            bgdvals = [-1.,-1]
          maxY = max( sigvals + bgdvals )  
          minY = min( sigvals + bgdvals )
          padding = np.abs(maxY - minY) * 0.25
          ylim0 = minY-2.5*padding 
          ylim1 = maxY + padding
          axbg.set_ylim( ylim0, ylim1)
          axsg.set_ylim( ylim0, ylim1)

          if Y in ylims.keys():
              ylims[Y].append(axbg) 
              ylims[Y].append(axsg)
          else:
              ylims[Y]= [ axbg, axsg ] 
          #axbg.set_ylim( 0., maxY + padding )
          #axsg.set_ylim( 0., maxY + padding ) 
  

          for tick in axsg.xaxis.get_major_ticks():
                tick.label.set_fontsize(7) 
                tick.label.set_rotation(70)
          for tick in axbg.xaxis.get_major_ticks():
                tick.label.set_fontsize(7) 
                tick.label.set_rotation(70)
          axbg.xaxis.get_major_ticks()[-1].set_visible(False)
          axsg.xaxis.get_major_ticks()[0].set_visible(False)

          for tick in axsg.yaxis.get_major_ticks():
                tick.label.set_fontsize(8)
          for tick in axbg.yaxis.get_major_ticks():
                tick.label.set_fontsize(8) 

          axbg.grid()
          axsg.grid()
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
              axbg.set_title('BASE')
              axsg.set_title('SIGNAL')
 
  
  for yk in ylims.keys():
    mins = [ a.get_ylim()[0] for a in ylims[yk] ] 
    maxs = [ a.get_ylim()[1] for a in ylims[yk] ] 
    for a in ylims[yk]:
      if max(maxs) > 10.:
        a.set_ylim( -5000., max(maxs) ) 
      else:
        a.set_ylim( min(mins), max(maxs) ) 

           
  scale = (cols**0.5)/3.0
  msscale = (cols**0.5)/3.5
                      
  data = np.array(data)
  bdata = np.array(bdata)

  if verbose:
    print "rows = ", rows
    print "cols = ", cols
    print "mainplotrows = ", mainplotrows
  if args.ratio == True:
    ax = plt.subplot( gs[ len(Yiter):len(Yiter)+mainplotrows, : (cols+1)/2 ] )
  else:
    ax = plt.subplot( gs[ len(Yiter):len(Yiter)+mainplotrows, :  ] )
  if args.ratio == True:
    ax2 = plt.subplot( gs[ len(Yiter):len(Yiter)+mainplotrows, (cols+1)/2 : ] )
  #ax = plt.subplot2grid( (rows,cols), (len(Yiter), 0), rowspan=mainplotrows, colspan=cols)

  if args.xlabels == None:
    xdat = data[:,0]
    bxdat = data[:,0]
  else:
    xdat = np.arange(0,data[:,1].size)
    bxdat = np.arange(0,bdata[:,1].size)

  def cnv_angle( num ):
    num = num - num % 1
    mm = np.nan
    if num == 1.:
      mm = -5.0 
    elif num == 2.:
      mm = -3.0 
    elif num == 3.:
      mm = 0. 
    elif num == 4.:
      mm = 1.5
    elif num == 5.:
      mm = 3.0
    elif num == 6.:
      mm = 4.5
    elif num == 7.:
      mm = 7.5
    else:
      if verbose:
        print "Error: Angle not defined."
      exit(1)
    return mm*4.37 # there are 4.37 mm/mrad

  v_cnv_angle = np.vectorize( cnv_angle)

  if args.anglex:
    xdat = v_cnv_angle( xdat )
    bxdat = v_cnv_angle( bxdat )
    


  ax.errorbar( xdat, data[:,1] , yerr=data[:,2], \
               capsize=0., elinewidth = 2.75*msscale ,\
               fmt='.', ecolor='blue', mec='blue', \
               mew=2.75*msscale, ms=11.0*msscale,\
               marker='o', mfc='lightblue', \
               label="signal")  

  print bdata
  if bdata.size > 0:
    ax.errorbar( bxdat, bdata[:,1] , yerr=bdata[:,2], \
                 capsize=0., elinewidth = 2.75*msscale ,\
                 fmt='.', ecolor='black', mec='black', \
                 mew=2.75*msscale, ms=11.0*msscale,\
                 marker='o', mfc='gray',\
                 label="baseline")

  if args.ratio == True:
    if bdata.size > 0:
      ax2.errorbar( xdat, data[:,1]/bdata[:,1] , yerr=data[:,2], \
                 capsize=0., elinewidth = 2.75*msscale ,\
                 fmt='.', ecolor='green', mec='green', \
                 mew=2.75*msscale, ms=11.0*msscale,\
                 marker='o', mfc='limegreen', \
                 label="signal/base")  
      maxY = max(data[:,1]/bdata[:,1])
      ax2.set_ylim( 0.9, maxY+0.05 )
      padding = (ax2.get_xlim()[1] - ax2.get_xlim()[0])*0.1
      ax2.set_xlim( ax2.get_xlim()[0] - padding ,ax2.get_xlim()[1] + padding)
      ax2.set_ylabel('A2/A1 (SIGNAL) / A2/A1 (BASE)')

    
    
    ax2.axhspan( 0.99, 1.01, facecolor='green', alpha=0.6, linewidth=0)
    for tick in ax2.xaxis.get_major_ticks():
      tick.label.set_fontsize(20.*scale)
    ax2.yaxis.tick_right()
    for tick in ax2.yaxis.get_major_ticks():
      tick.label2.set_fontsize(20.*scale)
    ax2.yaxis.set_label_position('right')

    ax2.grid()
    if args.xlabels != None:
      ax2.set_xticks( xdat )
      ax2.set_xticklabels( args.xlabels )
    
  
  allbdata = np.array(allbdata)
  allbdata = np.ma.masked_invalid( allbdata)
  allbdata = allbdata.compressed() 
  zero = np.mean( allbdata)
  zero_stderr = scipy.stats.sem( allbdata )
  
  ax.axhspan( zero-zero_stderr, zero+zero_stderr, facecolor='gray', alpha=0.6, linewidth=0)
  
  #ax.axvspan( 29.7, 33.7, facecolor='red', alpha=0.6, linewidth=0)
  #ax.axvspan( 105.7, 109.7, facecolor='red', alpha=0.6, linewidth=0)

  maxX = bxdat.max()   
  minX = bxdat.min()
  xpadding = np.abs( maxX - minX) * 0.2 / cols
  if xpadding ==  0.:
    if verbose:
      print 'Only one X point in set'
    xpadding = (ax.get_xlim()[1] - ax.get_xlim()[0])*0.05
  ax.set_xlim( minX - xpadding, maxX + 2.*xpadding )


  if args.xlabels == None:
    if args.anglex == False:
      ax.set_xlabel( args.XKEY, fontsize = 22.*scale )
    else:
      ax.set_xlabel( 'Bragg input (mrad)', fontsize = 22.*scale)
  ax.set_ylabel( 'A2/A1', fontsize = 22.*scale )

  for i,d in enumerate(data):
    sigi = data[i][1]
    sigierr = data[i][2]
    if bdata.size > 0: 
      bgdi = bdata[i][1]
      bgdierr = data[i][2]
    else:
      bdata = np.ones_like( data)
      bgdi = 1.0 
      bgdierr = 1.0

    s = ufloat( ( sigi, sigierr ) ) 
    b = ufloat( ( bgdi, bgdierr ) )

    a1 = ufloat( ( data[i][3], data[i][4] ))
    a2 = ufloat( ( data[i][5], data[i][6] ))

    ratio =  s / b

    sbase  = ufloat( ( bdata[i][1], bdata[i][2] ))
    a1base = ufloat( ( bdata[i][3], bdata[i][4] ))
    a2base = ufloat( ( bdata[i][5], bdata[i][6] ))
   
    # conciseout = 'a2 a1 a2/a1 a2base a1base a2/a1base
    CONCISEOUT += args.output + '\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\n'  % ( a2.nominal_value, a1.nominal_value, s.nominal_value, \
                                                              a2base.nominal_value, a1base.nominal_value, sbase.nominal_value ) 

    t0 = datetime.datetime( 2013, 07, 12, 22, 46 )
    deltasec = timestamp[i] - 3456532138.25
    delta = datetime.timedelta( seconds=deltasec)
    timestr = (t0+delta).strftime('%m/%d %H:%M') 
 
    meansig = a2 / a1 
    meansigstr = '%.2f' % meansig.nominal_value + '$\pm$%.2f' % meansig.std_dev() 

    #ax.text( xdat[i]+xpadding/5.*msscale, d[1], '%.2f\n' % ratio.nominal_value +'$\pm$%.2f' % ratio.std_dev(), \
    #         fontsize=26.*msscale , va='center') 
    #ax.text( xdat[i]+xpadding/5.*msscale, d[1]+0.005, '%.2f ' % s.nominal_value +'$\pm$%.2f' % s.std_dev(), \
    #         fontsize=16.*msscale , va='bottom')
 
    #ax.text( xdat[i]+xpadding/5.*msscale, d[1]-0.01, timestr + '\n' + meansigstr, \
    #         fontsize=16.*msscale , va='top') 

    ax.text( xdat[i]+xpadding/5.*msscale +0.5 , d[1]+0.005, timestr, \
             fontsize=14.*msscale , va='top')
 
  if args.xlabels != None:
    ax.set_xticks( xdat )
    ax.set_xticklabels( args.xlabels )

  for tick in ax.xaxis.get_major_ticks():
      tick.label.set_fontsize(20.*scale)
      #if args.xlabels != None:
      #  tick.label.set_rotation(15)
  for tick in ax.yaxis.get_major_ticks():
      tick.label.set_fontsize(20.*scale)
  
  ax.grid()
  #ax.legend(loc='best', numpoints=1,prop={'size':7.5*scale} ) 

  #gs.tight_layout(fig, rect=[0,0.03,1.,0.95])
 
  if args.output != None:
    if verbose:
      print "Saving figure to %s" % args.output
    fig.savefig( args.output, dpi=120 ) 
  else:
    plt.show() 
  
   
  print CONCISEOUT,
 

  

