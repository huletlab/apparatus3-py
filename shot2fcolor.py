#!/usr/bin/python

import numpy as np
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import sys

from configobj import ConfigObj

import argparse

# --------------------- DEFINITION OF COLOR MAPS --------------------#
 
grayscale_dict = {'blue': [(0.0, 1.0, 1.0), (0.125, 0.94117647409439087,
0.94117647409439087), (0.25, 0.85098040103912354,
0.85098040103912354), (0.375, 0.74117648601531982,
0.74117648601531982), (0.5, 0.58823531866073608, 0.58823531866073608),
(0.625, 0.45098039507865906, 0.45098039507865906), (0.75,
0.32156863808631897, 0.32156863808631897), (0.875,
0.14509804546833038, 0.14509804546833038), (1.0, 0.0, 0.0)],

    'green': [(0.0, 1.0, 1.0), (0.125, 0.94117647409439087,
    0.94117647409439087), (0.25, 0.85098040103912354,
    0.85098040103912354), (0.375, 0.74117648601531982,
    0.74117648601531982), (0.5, 0.58823531866073608,
    0.58823531866073608), (0.625, 0.45098039507865906,
    0.45098039507865906), (0.75, 0.32156863808631897,
    0.32156863808631897), (0.875, 0.14509804546833038,
    0.14509804546833038), (1.0, 0.0, 0.0)],

    'red': [(0.0, 1.0, 1.0), (0.125, 0.94117647409439087,
    0.94117647409439087), (0.25, 0.85098040103912354,
    0.85098040103912354), (0.375, 0.74117648601531982,
    0.74117648601531982), (0.5, 0.58823531866073608,
    0.58823531866073608), (0.625, 0.45098039507865906,
    0.45098039507865906), (0.75, 0.32156863808631897,
    0.32156863808631897), (0.875, 0.14509804546833038,
    0.14509804546833038), (1.0, 0.0, 0.0)]}

rainbow_dict = {'blue': [(0.0, 0.25882354378700256,
0.25882354378700256), (0.10000000000000001, 0.30980393290519714,
0.30980393290519714), (0.20000000000000001, 0.26274511218070984,
0.26274511218070984), (0.29999999999999999, 0.3803921639919281,
0.3803921639919281), (0.40000000000000002, 0.54509806632995605,
0.54509806632995605), (0.5, 0.74901962280273438, 0.74901962280273438),
(0.59999999999999998, 0.59607845544815063, 0.59607845544815063),
(0.69999999999999996, 0.64313727617263794, 0.64313727617263794),
(0.80000000000000004, 0.64705884456634521, 0.64705884456634521),
(0.90000000000000002, 0.74117648601531982, 0.74117648601531982), (1.0,
0.63529413938522339, 0.63529413938522339)],

    'green': [(0.0, 0.0039215688593685627, 0.0039215688593685627),
    (0.10000000000000001, 0.24313725531101227, 0.24313725531101227),
    (0.20000000000000001, 0.42745098471641541, 0.42745098471641541),
    (0.29999999999999999, 0.68235296010971069, 0.68235296010971069),
    (0.40000000000000002, 0.87843137979507446, 0.87843137979507446),
    (0.5, 1.0, 1.0), (0.59999999999999998, 0.96078431606292725,
    0.96078431606292725), (0.69999999999999996, 0.86666667461395264,
    0.86666667461395264), (0.80000000000000004, 0.7607843279838562,
    0.7607843279838562), (0.90000000000000002, 0.53333336114883423,
    0.53333336114883423), (1.0, 0.30980393290519714,
    0.30980393290519714)],

    'red': [(0.0, 0.61960786581039429, 0.61960786581039429),
    (0.10000000000000001, 0.83529412746429443, 0.83529412746429443),
    (0.20000000000000001, 0.95686274766921997, 0.95686274766921997),
    (0.29999999999999999, 0.99215686321258545, 0.99215686321258545),
    (0.40000000000000002, 0.99607843160629272, 0.99607843160629272),
    (0.5, 1.0, 1.0), (0.59999999999999998, 0.90196079015731812,
    0.90196079015731812), (0.69999999999999996, 0.67058825492858887,
    0.67058825492858887), (0.80000000000000004, 0.40000000596046448,
    0.40000000596046448), (0.90000000000000002, 0.19607843458652496,
    0.19607843458652496), (1.0, 0.36862745881080627,
    0.36862745881080627)]}

my_rainbow = mpl.colors.LinearSegmentedColormap('my_rainbow',rainbow_dict,256) 
my_grayscale = mpl.colors.LinearSegmentedColormap('my_grayscale',grayscale_dict,256)

# Definition of custom colormap by inverting an existing colormap
def invert_cmap( cm1 ):
    cdict = {}
    for primary in ('red','green','blue'):
        l=[]
        reversed = cm1._segmentdata[primary]
        reversed.reverse()
        for ituple in reversed:
           l.append(  ( 1-1*ituple[0],  ituple[2], ituple[1]) ) 
        cdict[primary]=l
    return colors.LinearSegmentedColormap('invert',cdict,1024) 

# Definition of custom colormap by taking part of an existing colormap
# c0 and cf must be between 0 and 1 and c0 < cf
def part_cmap( cm1, c0, cf ) :
    cdict = {}
    scale = cf-c0
    for primary in ('red','green','blue'):
        l=[]
	i0p = 0.
        i1p = 0.
        i2p = 0.
        for ituple in cm1._segmentdata[primary]:
            l0=len(l)
            if ituple[0] == 0 and c0==0.:
                l.append( (ituple[0], ituple[1], ituple[2]))
            elif ituple[0] > c0 and i0p <= c0 and len(l)==0:
                next = i2p + (c0-i0p)*(ituple[1]-i2p)/(ituple[0]-i0p)
                l.append( (0.0, next, next) )
            if ituple[0] >= cf and i0p < cf:
                prev = i2p + (cf-i0p)*(ituple[1]-i2p)/(ituple[0]-i0p)
                l.append( (1.0, prev, prev) )
                break
            if len(l) == l0 and len(l) != 0:
                l.append( ( i0p + (ituple[0]-i0p)/scale , ituple[1], ituple[2]))
            i0p=ituple[0]
            i1p=ituple[1]
            i2p=ituple[2]
        cdict[primary]=l
        
    return colors.LinearSegmentedColormap('part',cdict,1024) 
          
           

# Definition of custom colormap by merging two colormaps
# uses cm1 from 0 to cutoff and cm2 from cutoff to 1
def merge_cmaps( cm1, cm2, cutoff):
    cdict = {}
    for primary in ('red','green','blue'):
        l=[]
        for ituple in cm1._segmentdata[primary]:
            l.append( ( cutoff*ituple[0], ituple[1], ituple[2]) )
        for ituple in cm2._segmentdata[primary]:
            l.append( ( cutoff + (1-cutoff)*ituple[0], ituple[1], ituple[2]) )
        cdict[primary]=l
    return colors.LinearSegmentedColormap('merged', cdict,1024)





# --------------------- MAIN CODE  --------------------#


if __name__ == "__main__":
  parser = argparse.ArgumentParser('shot2fcolor.py')
 
  parser.add_argument('SHOT', action="store", type=int, help='shot number')
  parser.add_argument('-r', action="store", dest='ROI', type=str, help='X0,Y0,XW,YW region of interest')
  parser.add_argument('-c', action="store", dest='CRANGE', type=str, help='CMIN,CMAX range of cmap')
  parser.add_argument('--xc', action="store", dest='CXOVER', type=str, help="crossover c between colormaps")
  parser.add_argument('-d', action = "store", dest='DIR', type=str, help="path of directory for output png file")
  parser.add_argument('--sort', action = "store", dest='SECKEY', type=str, help="value of this SEC:KEY in the report is prepended to the file name")
  parser.add_argument('--dpi', action = "store", dest='DPI', type=int, help="dots per inch for png output")
  parser.add_argument('--ppi', action = "store", dest='PPI', type=int, help="camera pixels per inch for png output")

  args = parser.parse_args()

  print type(args)
  print args

  shotnum = "%04d" % int(args.SHOT)
  columnfile = shotnum + "_column.ascii"
  column = np.loadtxt(columnfile)

  inifile = "report" + shotnum + ".INI"
  report = ConfigObj(inifile)

  if args.ROI:
    X0 = float(args.ROI.split(',')[0])
    Y0 = float(args.ROI.split(',')[1])
    XW = float(args.ROI.split(',')[2])
    YW = float(args.ROI.split(',')[3])
    column = column[Y0:Y0+YW, X0:X0+XW]
  else:
    X0 = 0.
    Y0 = 0.
    XW = column.shape[0]
    YW = column.shape[1] 
  print column.shape

  if args.CRANGE:
    cmin = float(args.CRANGE.split(',')[0])
    cmax = float(args.CRANGE.split(',')[1])
  else:
    cmin = column.min()
    cmax = column.max() 
  
  if args.CXOVER:
    cutoff = float( args.CXOVER )
  else:
    cutoff = cmin

  if args.DPI:
    dpi = args.DPI
  else:
    dpi = 120

  if args.PPI:
    ppi = args.PPI
  else:
    ppi = 96
 
  partgray = part_cmap( cm.gray, 0.0, 0.8)
  invertgray = invert_cmap( partgray )
  partspectral = part_cmap( cm.spectral, 0.2, 0.95)  

  merged=merge_cmaps( part_cmap( cm.gray, 0.2, 0.75) , part_cmap(cm.spectral, 0., 0.5) , (cutoff-cmin)/(cmax-cmin) )
#merged=merge_cmaps( my_grayscale, my_rainbow, (cutoff-cmin)/(cmax-cmin) )
#merged=merge_cmaps( cm.gray, cm.spectral, (cutoff-cmin)/(cmax-cmin) )

  fig = plt.figure(figsize=(YW/ppi,XW/ppi))
  ax = fig.add_axes([0.0,0.04,1.0,0.92],frameon=True)

  ax.set_yticklabels([])
  ax.set_xticklabels([])

  im = ax.imshow(column, cmap=partspectral, vmin=cmin,vmax=cmax, origin='lower')
  ax = im.get_axes()

  if args.DIR:
     pngpath = args.DIR + shotnum + '_shot2fcolor.png' 
     txtpath = args.DIR + shotnum + '_shot2fcolor.txt' 
  else:
     pngpath = shotnum + '_shot2fcolor.png' 
     txtpath = shotnum + '_shot2fcolor.txt' 
     
  plt.savefig(pngpath, dpi=dpi)
  np.savetxt(txtpath, column)
   

  exit(1) 


# --------------------  START OF PLOTTING PART --------------------#

#for i in dir(ax):
#    print i

#mpl.rc('mathtext',default='regular')

#Scale label
#ax.text(10,540,r"$100\ \mu\mathrm{m}$",fontsize=16)

#Paramters label 
#sec = sys.argv[SECKEY].split(':')[0]
#key = sys.argv[SECKEY].split(':')[1]
#val = report[sec][key]
#labeltext="%s:%s=%s" % (sec,key,val)
#ax.text(10,-2,labeltext, fontsize=8)

#Other labels
#labeltext="%s:%s=%.2e" % ('CPP','nfit',float(report['CPP']['nfit']))
#ax.text(10, -9, labeltext, fontsize=8)

#labeltext="%s:%s=%.2e" % ('ANDOR','phcdet',float(report['ANDOR']['phcdet']))
#ax.text(10, -16, labeltext, fontsize=8)

#labeltext="%s:%s=%.2e" % ('ODT','odttof',float(report['ODT']['odttof']))
#ax.text(10, -23, labeltext, fontsize=8)


#scalebar = patches.Rectangle((10,500),31.25,6,ec='black', fc='black')
#ax.add_patch(scalebar)

#for i in dir(plt):
#    print i   

#plt.colorbar()
#plt.show()

