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

iniPath = os.path.abspath(__file__).replace("showlatticealign.py","autoAlignGr.ini")

inifile = ConfigObj(iniPath)
# The slop for 
pairs = ["GR1H","GR1V","GR2H","GR2V","GR3H","GR3V"]

picoslope=[]
grEr=[]
for p in pairs:
	temp=[ float(i) for i in inifile["SLOPE"][p]]
	if temp[1]==0:
		picoslope.append(temp[0])
	else:
		picoslope.append(temp[1])

	temp=[ float(i) for i in inifile["OVERRIDE_GR"][p]]
	grEr.append(temp[-1])


#picoslopeOLD = [0.97,-1.95,-1.39,-1.3,-1.9,-0.95]

#print picoslopeOLD
#print picoslope
#print grEr
#exit()

#exit()




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
	print beams
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
                d0 = abs(irpair[pair][0][0]-irpair[apair][0][0])
                d1 = abs(irpair[pair][0][1]-irpair[apair][0][1])
                if d0>wc: 
			warning = 1
			warningstring = warningstring + pair +" and " +apair+" separate in 0 by %.2f pixel.\n"%d0 
                if d1>wc: 
			warning = 1
			warningstring = warningstring + pair +" and " +apair+" separate in 1 by %.2f pixel.\n"%d1
		#if ( d0>wc) | (d1 >wc):
		#	warning = 1
		#	warningstring = warningstring + "Warning! " + pair +" and " +apair+" separate by more than %.1f pixel.\n"%wc 
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
  rowlabel = []
  pairs =['1+2IR %.1fEr'%(grEr[0]), \
	     '1+3IR %.1fEr'%(grEr[1]), \
	     '1+1IR %.1fEr'%(grEr[2]), \
	     '2+3IR %.1fEr'%(grEr[3]), \
	     '1+3IR %.1fEr'%(grEr[4]), \
	     '2+3IR %.1fEr'%(grEr[5])]
  collabel = ["Goal","Goal Waist","Position","Delta Pos","Delta Waist","S Pico","Pair"]
  #colcolor = ['grey','grey','grey','grey']
  rowcolor =[]
 
  for k,picoslope,pair in zip(sorted(grdata.keys()),picoslope,pairs):
        rowlabel.append(k)
	c = [ center[1]-center[0] for center in zip(grdata[k][0][0],grdata[k][1][0])]
	w = [ waist[1]-waist[0] for waist in zip(grdata[k][0][1],grdata[k][1][1])]
	w_goal = [ waist[0] for waist in zip(grdata[k][0][1],grdata[k][1][1])]
        suggestpico = [ int(c[0 if((k=="GR1V")|(k=="GR2V")) else 1]*picoslope)]
	rowcolor.append('g' if (abs(c[0])<1) & (abs(c[1])<1)else 'r' )
        tabledata.append(['( %.2f , %.2f )'%(tuple(grdata[k][0][0]))]+['( % .2f , % .2f )'%(tuple(w_goal))]+['( %.2f , %.2f )'%(tuple(grdata[k][1][0]))]+['( % .2f , % .2f )'%(tuple(c))]+ ['( % .2f , % .2f )'%(tuple(w))] +suggestpico+ [pair])
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
	colWidths=[0.1]*7,
	rowLabels=rowlabel,
	colLabels=collabel,
	rowColours=rowcolor,
	loc='top')

  table_cells=table.properties()['child_artists']
  table_cells=[ table._cells[k] for k in table._cells.keys()]

  for i,cell in enumerate(table_cells):
      cell.set_fontsize(40)
      cell.set_width(0.15 )
  
  #table.set_fontsize(36)
  #table.scale(1.5,1.5)
  
  circle = plt.Circle((0,0),radius=1.0,color='g',alpha=0.5)
  axes.add_artist(circle)
  plt.subplots_adjust(top=0.8,bottom=0.1,left=0.27,right=0.8)





  if args.output != None:
	print args.output
  	if not os.path.exists(args.output):
		os.makedirs(args.output)
	print "Saving figure to %s" % args.output
	fig.savefig( args.output +  "lattice_align_"+rangestr+".png",  dpi=240 ) 
  else:
  	if not os.path.exists("plots"):
		os.makedirs("plots")
    	fig.savefig( "plots/lattice_align_"+rangestr+".png", dpi=240 ) 
