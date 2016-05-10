#!/usr/bin/python

import sys
import argparse
import os
import glob
from configobj import ConfigObj

from math import copysign

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#import itertools

sys.path.append('/lab/software/apparatus3/py')

#import falsecolor
#import gaussfitter
import qrange
import statdat
import scipy

#from uncertainties import ufloat

import pprint

# --------------------- MAIN CODE  --------------------#

if __name__ == "__main__":
  parser = argparse.ArgumentParser('showlatticealign.py')

  parser.add_argument('--range', action = "store", \
         help="range of shots to be used.")

  parser.add_argument('--output', action="store", \
         help="optional path of png to save figure") 

  args = parser.parse_args()

  #print args
 
  rangestr = args.range.replace(':','to')
  rangestr = rangestr.replace(',','_')
  shots = qrange.parse_range(args.range)
  if len(shots)!=9:
	sys.exit("Must have 9 shots in range!")
  
  # The key for the dictionary is (ir1,ir2,ir3,gr1,gr2,gr3). 

  shotsdata = {(1,1,0,0,0,0):[(0,0),(0,0)],\
	(1,1,0,1,0,0):[(0,0),(0,0)],\
	(1,1,0,0,1,0):[(0,0),(0,0)],\
	(1,0,1,0,0,0):[(0,0),(0,0)],\
	(1,0,1,1,0,0):[(0,0),(0,0)],\
	(1,0,1,0,0,1):[(0,0),(0,0)],\
	(0,1,1,0,0,0):[(0,0),(0,0)],\
	(0,1,1,0,1,0):[(0,0),(0,0)],\
	(0,1,1,0,0,1):[(0,0),(0,0)]}
  
  keys = ["DIMPLE:ir1",\
	"DIMPLE:ir2",\
	"DIMPLE:ir3",\
	"DIMPLE:gr1",\
	"DIMPLE:gr2",\
	"DIMPLE:gr3",\
	"CPP:ax0c",\
	"CPP:ax1c",\
	"CPP:ax0w",\
	"CPP:ax1w"]

  for s in shots:
	report = ConfigObj( 'report'+s+'.INI')  
	temp =[]
	for k in keys:
		temp.append(qrange.evalstr( report, k  ))
	beams = tuple(temp[0:6])
	center = tuple(temp[6:8])
	waist = tuple(temp[8:10])

	if (beams in shotsdata.keys()):
		shotsdata[beams] = [center,waist]
	else:
		sys.exit("Unexpect IR GR pair condition.")
  grdata = {'GR1V':[shotsdata[(1,0,1,0,0,0)],shotsdata[(1,0,1,1,0,0)]],\
	'GR1H':[shotsdata[(1,1,0,0,0,0)],shotsdata[(1,1,0,1,0,0)]],\
	'GR2V':[shotsdata[(0,1,1,0,0,0)],shotsdata[(0,1,1,0,1,0)]],\
	'GR2H':[shotsdata[(1,1,0,0,0,0)],shotsdata[(1,1,0,0,1,0)]],\
	'GR3V':[shotsdata[(0,1,1,0,0,0)],shotsdata[(0,1,1,0,0,1)]],\
	'GR3H':[shotsdata[(1,0,1,0,0,0)],shotsdata[(1,0,1,0,0,1)]]}
  
  #Check lattice pair condition
  irpair = {"IR12":shotsdata[(1,1,0,0,0,0)],"IR23":shotsdata[(0,1,1,0,0,0)],"IR13":shotsdata[(1,0,1,0,0,0)]}
  warning = 0 
  warningstring =""
  wc = 1.0
  keys = irpair.keys()
  for pair in keys:
     for apair in keys:
	if pair != apair:
		if (abs(irpair[pair][0][0]-irpair[apair][0][0]) >wc)|(abs(irpair[pair][0][1]-irpair[apair][0][1]) >wc):
			warning = 1
			warningstring = warningstring + "Warning! " + pair +" and " +apair+" seperate by more than %.1f pixel.\n"%wc 
     keys.pop(0)
  pprint.pprint( grdata )
  
 
  fig = plt.figure(  )
  axes = fig.add_subplot(111)
  
  allx =[]
  ally =[]
  colors = ["r","b","k"]
  textpos = 1.0
  textoff = 0.5
  tabledata =[]
  rowlabel =[]
  collabel = ["Goal","Goal Waist","Position","Delta Pos","Delta Waist","Suggest Pico"]
  #colcolor = ['grey','grey','grey','grey']
  # The slop for GR1H,GR1V,GR2H,GR2V,GR3H,GR3V
  picoslope = [4.05,-1.95,-1.39,-1.3,-1.9,-0.95]
  rowcolor =[]
 
  for k,picoslope in zip(sorted(grdata.keys()),picoslope):
        rowlabel.append(k)
	c = [ center[1]-center[0] for center in zip(grdata[k][0][0],grdata[k][1][0])]
	w = [ waist[1]-waist[0] for waist in zip(grdata[k][0][1],grdata[k][1][1])]
	w_goal = [ waist[0] for waist in zip(grdata[k][0][1],grdata[k][1][1])]
        suggestpico = [ int(c[0 if((k=="GR1V")|(k=="GR2V")) else 1]*picoslope)]
        suggestpico = [ int(c[0 if((k=="GR1V")|(k=="GR2V")) else 1]*picoslope)]
	rowcolor.append('g' if (abs(c[0])<1) & (abs(c[1])<1)else 'r' )
        tabledata.append(['( %.2f , %.2f )'%(tuple(grdata[k][0][0]))]+['( % .2f , % .2f )'%(tuple(w_goal))]+['( %.2f , %.2f )'%(tuple(grdata[k][1][0]))]+['( % .2f , % .2f )'%(tuple(c))]+ ['( % .2f , % .2f )'%(tuple(w))] +suggestpico)
	allx.append(c[0])
	ally.append(c[1])
	ls = 'dotted' if k[-1]=='H' else 'solid'
	color = colors[int(k[2])-1]
	axes.arrow(0,0,c[0],c[1],head_width=0.2,head_length=0.1,fc=color,ec=color, ls = ls )
	axes.text(c[0]*textpos+copysign(textoff,c[0]),c[1]*textpos+copysign(textoff,c[1]),k,color=color,size=12)
  
 
  limit = max ( [ abs(j) for j in allx+ally] + [5] ) 
  if warning: 
  	axes.text(-limit-.5,-limit-.5,warningstring,color='r',size=10)
  axes.set_xlim([-limit-1,limit+1])	
  axes.set_ylim([-limit-1,limit+1])	
  print tabledata 
  table = axes.table(cellText=tabledata,
	colWidths=[0.10]*6,
	rowLabels=rowlabel,
	colLabels=collabel,
	rowColours=rowcolor,
	loc='top')

  table_cells=table.properties()['child_artists']
 
  for cell in table_cells:
      cell.set_fontsize(36)
      cell.set_width(0.2)
  
  #table.set_fontsize(36)
  #table.scale(1.5,1.5)
  
  circle = plt.Circle((0,0),radius=1.0,color='g',alpha=0.5)
  axes.add_artist(circle)
  plt.subplots_adjust(top=0.8,bottom=0.1,left=0.27,right=0.8)
 
  if args.output != None:
    print "Saving figure to %s" % args.output
    fig.savefig( args.output, dpi=240 ) 
  else:
    fig.savefig( "plots/lattice_align2_"+rangestr+".png", dpi=240 ) 
