#!/usr/bin/python

import argparse
import fitlibrary
import numpy
import datetime

import matplotlib.pyplot as plt
import matplotlib
import sys
sys.path.append('/lab/software/apparatus3/bin/py')

import statdat

# --------------------- MAIN CODE  --------------------#


if __name__ == "__main__":
  parser = argparse.ArgumentParser('plotbeamprofiles.py')
  parser.add_argument('wavelength', action="store", type=int, help='wavelength of beam')
  parser.add_argument('datfiles', nargs='*', help='list of dat files to fit')

  args = parser.parse_args()
  #print type(args)
  #print args
  if args.wavelength == 1070:
    fit = fitlibrary.fitdict['Beam1070']
  
  else:
    print "  ERROR: unrecognzed wavelength. Program will exit"


  matplotlib.rcdefaults()
  figw = 12.0
  figh = 14.5
  fig = plt.figure( figsize=(figw,figh) )
  ax1 = fig.add_axes( [0.13,0.6,0.73,0.37])
  ax1b = ax1.twinx()
  ax2 = fig.add_axes( [0.15,0.08,0.69,0.28])
  ax3 = ax2.twinx()

  colors = ['red','green', 'blue', 'black', 'magenta', 'cyan', 'yellow', 'orange', 'firebrick', 'steelblue']

  post = []
#  HposRef = None
#  VposRef = None
  
  for i,dat in enumerate(args.datfiles):
    print i    
    print "Fitting %s" % dat
    d = numpy.loadtxt( dat)
    p0 = [ 70., 250.]
    Hpfit, Herror = fitlibrary.fit_function( p0, d[:, [0,1]], fit.function)  
    Vpfit, Verror = fitlibrary.fit_function( p0, d[:, [0,2]], fit.function)  

    Hleg = "%s\nwH = %.2f um at %.0f MIL" % ( dat, Hpfit[0], Hpfit[1] ) 
    Vleg = "wV = %.2f um at %.0f MIL" % ( Vpfit[0], Vpfit[1] )
   
    astigmatism = Hpfit[1] - Vpfit[1] 
    postdat = numpy.array( [[ i, Hpfit[0], Vpfit[0], astigmatism]] )
    ax2.plot( postdat[:,0], postdat[:,1], 'o', color=colors[i], markeredgewidth=0.3, markersize=12)
    ax2.plot( postdat[:,0], postdat[:,2], 'x', color=colors[i], markeredgewidth=1.0, markersize=12)
    ax3.plot( postdat[:,0], postdat[:,3], '^', markerfacecolor="None", markeredgecolor=colors[i], markeredgewidth=1.0, markersize=12)
    
   
    print "\t" + Hleg
    print "\t" + Vleg 

    HfitX, HfitY = fitlibrary.plot_function( Hpfit, d[:,0], fit.function)
    VfitX, VfitY = fitlibrary.plot_function( Vpfit, d[:,0], fit.function)

    msize=8
    ax1.plot( d[:,0], d[:,1], 'o', color = colors[i], markersize=msize, markeredgewidth=0.3, label=Hleg)
    ax1.plot( d[:,0], d[:,2], 'x', color = colors[i], markersize=msize, markeredgewidth=1.0, label=Vleg )

    ax1.plot( HfitX, HfitY, linestyle = '-', color = colors[i])
    ax1.plot( VfitX, VfitY, linestyle = '-', color = colors[i])
    
    
    Hlegb = "%s\nX Position of Waist" %dat
    Vlegb = "Y Position of Waist" 

    camPixSize = 4.65 
    
    msizeb= 8

    if i == 0:
        HposRefStat = statdat.statdat( d[:,[0,6]] , 0, 1)
        HposRef = HposRefStat[:,1] * camPixSize
        VposRefStat = statdat.statdat( d[:,[0,7]] , 0, 1)
        VposRef = VposRefStat[:,1] * camPixSize
        print HposRef
        print VposRef

    if i > 0:
        HposStat = statdat.statdat( d[:,[0,6]] , 0, 1)
        Hpos = HposStat[:,1] * camPixSize
        VposStat = statdat.statdat( d[:,[0,7]] , 0, 1)
        Vpos = VposStat[:,1] * camPixSize
        ax1b.plot( HposStat[:,0], Hpos - HposRef, '->', alpha=0.3,color = colors[i], markersize=msizeb, markeredgewidth=0.3, label=Hlegb)
        ax1b.plot( HposStat[:,0], Vpos - VposRef, '-<', alpha=0.3,color = colors[i], markersize=msizeb, markeredgewidth=1.0, label=Vlegb)


  ax1.legend(loc='upper left', bbox_to_anchor = (0.0,-0.06), prop={'size':10}, numpoints=1)
  ax1b.legend(loc='upper right', bbox_to_anchor = (1.0,-0.06), prop={'size':10}, numpoints=1)
  ax1.yaxis.set_major_formatter( matplotlib.ticker.FormatStrFormatter(r'%.1f'))
  ax1.xaxis.set_major_formatter( matplotlib.ticker.FormatStrFormatter(r'%d'))

  fsize = 18
  for tick in ax1.xaxis.get_major_ticks():
    tick.label.set_fontsize(fsize)
  for tick in ax1.yaxis.get_major_ticks():
    tick.label.set_fontsize(fsize)

  ax1.set_ylim(57,92)
  ax1.set_xlim(0,500)

  ax1.spines["bottom"].set_linewidth(2)
  ax1.spines["top"].set_linewidth(2)
  ax1.spines["left"].set_linewidth(2)
  ax1.spines["right"].set_linewidth(2)

  ax1.set_xlabel(r"Z (MIL)", fontsize=fsize, labelpad=16)
  ax1.set_ylabel(r"1/e^2 radius (um)", fontsize=fsize, labelpad=20)
  ax1b.set_ylabel('Delta Position on Camera\nwith respect to red (um)', fontsize=fsize, labelpad=25, ha = 'center')

  ax2.set_xlabel(r"File number", fontsize=fsize/1.0, labelpad=16)
  ax2.set_ylabel('Beam waist (um)\ncircles (H)  and crosses (V)', fontsize=fsize/1.0, labelpad=30, ha = 'center')
  ax3.set_ylabel('Astigmatism (MIL)\ntriangles (H - V)', fontsize=fsize/1.0, labelpad=30, ha = 'center')
  ax2.xaxis.set_major_formatter( matplotlib.ticker.FormatStrFormatter(r'%d'))
  ax2.xaxis.set_major_locator( matplotlib.ticker.MultipleLocator( 1.0) )
  ax2.set_xlim(-0.2, len(args.datfiles)- 0.8 )
  fsize = 18
  for tick in ax2.xaxis.get_major_ticks():
    tick.label.set_fontsize(fsize)
  for tick in ax2.yaxis.get_major_ticks():
    tick.label.set_fontsize(fsize)
  ax2.spines["bottom"].set_linewidth(2)
  ax2.spines["top"].set_linewidth(2)
  ax2.spines["left"].set_linewidth(2)
  ax2.spines["right"].set_linewidth(2)
  ax2.set_ylim(56,70) 

  output = datetime.datetime.now().strftime("%Y%m%d_%H%M%S.png")
  print output
  #fig.savefig( "debug.png" , dpi=140)
  fig.savefig( output , dpi=140)



  exit(1)
