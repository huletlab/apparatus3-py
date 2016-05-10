#!/usr/bin/python

import sys
from numpy import loadtxt
from numpy import amin, amax

import os
import glob

import matplotlib.pyplot as plt
import matplotlib

import argparse

if __name__ == "__main__":

  parser = argparse.ArgumentParser('inspectAz_ascii')

  #shotnum is used for the filename prefix 
  parser.add_argument('shotnum', action="store", type=int, help="shot number") 

  parser.add_argument('--andor2', action="store_true", dest='andor2', help='set andor2 flag')  
  parser.add_argument('--pubqual', action="store_true", dest='pubqual', help='set publication quality flag')  

  parser.add_argument('--fermi2d', action="store", dest='fermi2d', type=float,help='set value for 2D T/TF') 
  parser.add_argument('--fermiazimuth', action="store", dest='fermiazimuth', type=float, help='set value for azimuth  T/TF') 
  
 
  args = parser.parse_args() 
  shotnum = args.shotnum 
  shotnum = '%04d' % shotnum 

  # This program should be called as 
  # inspectAz_ascii.py  shot prefix
  #if len(sys.argv) > 2:
  #  prefix = sys.argv[2]
  #else:
  #  prefix = ''

  if args.andor2: 
     list = glob.glob(  os.getcwd() + '/' + shotnum + '_andor2' + '*.AZASCII' ) 
  else:
     list = glob.glob(  os.getcwd() + '/' + shotnum + '*.AZASCII' ) 
  
  dats =[]
  fits =[]
  other=[]
  for file in list:
    label = file.rsplit('_')[-1].split('.')[0]
#    print file
    if '_dat' in file:
      dats.append( [loadtxt (file),label] )
    elif '_fit' in file:
      fits.append( [loadtxt (file),label] )
    else:
      other.append( [loadtxt (file),label] ) 


  figw=9.
  figh=6.
  fig = plt.figure( figsize=(figw,figh))
  ax1 = fig.add_axes( [0.15, 0.15, 0.75,0.75])
  
  ax1.axvline(x=0, linewidth=1.0, color='black')
  ax1.axhline(y=0, linewidth=1.0, color='black')
 
# Data and fits for publication quality figures
  pubqualData = []
  pubqualFits = []
 
#  print "Processing dats..."
  for d in dats: 
#    print d[1]
    alpha = 1.0
    color = 'green'
    marker = '.'
    markersize = 8
    if 'Icut' in d[1]:
      alpha = 0.25
      color = 'red'
    if 'Jcut' in d[1]:
      alpha = 0.25
      color = 'blue'
    if 'Azimuth' in d[1] and not 'All' in d[1]:
      marker = 'D'
      markersize = 6
      if args.pubqual:
         pubqualData.append(d) 
    if 'Azimuth' in d[1]:
      alpha = 0.5
    ax1.plot( d[0][:,0], d[0][:,1], marker, color=color, \
              markersize=markersize, markeredgewidth=0.1, markeredgecolor='black', \
              label = d[1], alpha = alpha )

#  print "Processing fits..." 
  for f in fits: 
#    print f[1]
    alpha = 1.0
    color = 'black'
    marker = '-'
    markersize = 8
    lw = 1
    if f[1] == 'fit2DFermi' or f[1] =='fit2DGauss' or f[1] == 'fitAzimuth':
       pubqualFits.append(f)
 
    if 'Azimuth' in f[1] and not 'ZeroT' in f[1]:
      color = 'red'
    if 'AzimuthZeroT' in f[1]:
      color = '#35C6CD'
      alpha = 0.75
      lw = 2.5
    if '2DFermi' in f[1]:
      color = '#FFC000'
    ax1.plot( f[0][:,0], f[0][:,1], marker, color=color, \
              markersize=markersize, markeredgewidth=0.1, \
              markeredgecolor='black', \
              linewidth = lw, label = f[1], alpha = alpha )
  for o in other:
    print o[1]

  prop = matplotlib.font_manager.FontProperties(size=12)
  ax1.legend( prop=prop,loc= 'upper right', bbox_to_anchor=(0.97,0.97),numpoints=1, handlelength=0.6, handletextpad=0.5)

  extratxt = ''
  if args.fermi2d is not None:
      extratxt += r'$T/T_{\mathrm{F,2D}}=%.2f$'%args.fermi2d
  if args.fermiazimuth is not None:
      extratxt += '\n'
      extratxt += r'$T/T_{\mathrm{F,AZ}}=%.2f$'%args.fermiazimuth
  if extratxt != '':
      ax1.text(0.05, 0.95, extratxt, transform=ax1.transAxes, fontsize=12,\
               verticalalignment='top')

  fig.savefig( shotnum+'_Azimuth_inspect.png', dpi=140) 

  #Down here make a nice plot if publication or talk quality is required
  if not args.pubqual:
     exit(0)

  from matplotlib import rc
  rc('font',**{'family':'serif'})
  fig = plt.figure( figsize=(4.0,3.5))
  ax1 = fig.add_subplot(111)


  for f in pubqualFits:
    if f[1] == 'fit2DGauss':
      alpha = 1.0
      color = 'black'
      marker = '-'
      markersize = 8
      lw = 1 
      label = '2D Gaussian fit'
      label = ''
    
    if f[1] == 'fit2DFermi':
      if not args.fermi2d:
        continue
      alpha = 1.0
      color = '#FFC000'
      marker = '-'
      markersize = 8
      lw = 1 
      label = '2D fit'
      if args.fermi2d:
        label = label + ' $T/T_{F}=%.2f$' % args.fermi2d

    if f[1] == 'fitAzimuth':
      if not args.fermiazimuth: 
        continue
      alpha = 1.0
      color = 'red'
      marker = '-'
      markersize = 8
      lw = 1 
      label = 'Azimuthal fit'
      if args.fermiazimuth:
        label = label + ' $T/T_{F}=%.2f$' % args.fermiazimuth
        label = ' $T/T_{F}\ =\ 0.09$'

    ax1.plot( f[0][:,0], f[0][:,1], marker, color=color, \
              markersize=markersize, markeredgewidth=0.1, \
              markeredgecolor='black', \
              linewidth = lw, label = label, alpha = alpha )
  
  xlim0 = 0.
  xlim1 = 100. 

  for d in pubqualData: 
    alpha = 0.6
    color = 'red'
    marker = '.'
    markersize = 8
    
    xdat = d[0][:,0]
    xlim0 = amin(xdat)
    xlim1 = amax(xdat)
   
    ax1.plot( d[0][:,0], d[0][:,1], marker, color=color, \
              markersize=markersize, markeredgewidth=0.1, markeredgecolor='black', \
              alpha = alpha )
   
    
  prop = matplotlib.font_manager.FontProperties(size=12)
  ax1.legend( prop=prop,loc= 'upper right', bbox_to_anchor=(0.97,0.97),numpoints=1, handlelength=0.6, handletextpad=0.5) 
  
  ax1.set_ylim( -10., None)
  if xlim1 > 45.: 
    xlim1 = 45. 
  ax1.set_xlim( xlim0, xlim1)
  ax1.set_yticklabels([]) 
  ax1.grid()
  
   
  ax1.set_xlabel('Distance from center (pixels)')
  ax1.set_ylabel('Column density', multialignment='center')

  fig.tight_layout()
  fig.savefig( shotnum+'_Azimuth_inspect_pubquality.png', dpi=140) 
   
   
 
