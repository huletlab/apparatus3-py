#!/usr/bin/python

import sys
from numpy import loadtxt

import os
import glob

import matplotlib.pyplot as plt
import matplotlib





if __name__ == "__main__":
  # This program should be called as 
  # inspectAz_ascii.py  shot prefix
  shotnum = '%04d' % int(sys.argv[1])
  if len(sys.argv) > 2:
    prefix = sys.argv[2]
  else:
    prefix = ''

  if prefix == 'andor2':
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
  
#  for d in dats:
 
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

  fig.savefig( shotnum+'_'+prefix+'Azimuth_inspect.png', dpi=140) 
    
    

   
   
 
