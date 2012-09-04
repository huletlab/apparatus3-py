#!/usr/bin/python

import numpy as np
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def savepng( data , cmin, cmax , colormap, id, dpi, origin='lower'): 
   """Save falsecolor png file of data"""
   pixel_per_inch=32.
   #    fig = plt.figure(figsize=(data.shape[0]/pixel_per_inch,data.shape[1]/pixel_per_inch+0.5))
   fig = plt.figure()
   ax = fig.add_axes([0.0,0.05,1.0,0.85],frameon=True)

   im = ax.imshow(data, cmap=colormap, vmin=cmin,vmax=cmax, origin=origin)
   ax = im.get_axes()

   pngpath = id + '_falsecolor.png' 
   plt.savefig(pngpath, dpi=dpi)
   return pngpath

def inspecpng( imglist , inspec_row, inspec_col, cmin, cmax, colormap, id, dpi, origin='lower'):
   """Makes an inspector plot (falsecolor plus cuts) for the list of images, the first image
      is always the data and the rest are treated as fits"""
   shape = imglist[0].shape
   for img in imglist:
     if img.shape != shape:
       print " --->  ERROR:  falsecolor.inspecpng  cannot handle images of different shapes"
       exit(1)
   # The first image in the list is used for the false color plot
   figw = 6.  #fig width in inches

   axw = 0.50  #fraction of fig for axes width
   axh = 0.60  #fraction of fig for axes width
    
   figh = axw*figw  *shape[0] / shape[1] / axh # fig height in inches
   #axh = axw*figw  *shape[0] / shape[1] / figh #fraction of fig for axes height

   #print axw, axh
   #h = w/aspect
   fig = plt.figure(figsize=(figw,figh))

   xstart = 0.1
   xgap = 0.14
   
   ystart = 0.1
   ygap = 0.14
   
   xs = 0.4*axw
   ys = 0.4*axh

   ax = fig.add_axes([xstart,ystart, axw, axh],frameon=True)
   im = ax.imshow(imglist[0], cmap=colormap, vmin=cmin,vmax=cmax, origin=origin)


   axFIT = fig.add_axes( [xgap+axw, ygap+axh, xs, ys], frameon=True)
   axFIT.imshow( imglist[1], cmap=colormap, vmin=cmin, vmax=cmax, origin=origin)
   axFIT.xaxis.set_ticklabels([])
   axFIT.yaxis.set_ticklabels([])
 
   alphacross = 0.5
   alphadata = 0.35 
   
   ax.axhline( inspec_row, linewidth=0.5, color='blue', alpha=alphacross)
   axROW = fig.add_axes([xstart,ygap+axh, axw, ys], frameon=True)
   axROW.set_xlim( 0, len(imglist[0][ inspec_row, :])-1)
   axROW.plot( imglist[0][ inspec_row, :] , color='blue', alpha=alphadata)
   for img in imglist[1:]:
     axROW.plot( img[ inspec_row, :] , color='blue')
   axROW.xaxis.set_ticklabels([]) 

   ax.axvline( inspec_col, linewidth=0.5, color='red', alpha=alphacross)
   axCOL = fig.add_axes([xgap+axw,ystart, xs , axh], frameon=True)
   xarray = np.arange( len( imglist[0][:, inspec_col]) )
   axCOL.set_ylim( len( imglist[0][ :, inspec_col])-1, 0)
   axCOL.plot( imglist[0][ : , inspec_col], xarray , color='red', alpha=alphadata)
   for img in imglist[1:]:
     axCOL.plot( img[ : , inspec_col], xarray , color='red')
   axCOL.yaxis.set_ticklabels([]) 
   labels = axCOL.get_xticklabels()
   for label in labels:
     label.set_rotation(-90)

   ax.set_xlim ( 0, len(imglist[0][ inspec_row, :])-1 )
   ax.set_ylim ( len(imglist[0][ :, inspec_col])-1, 0)
   

   pngpath = id + '_inspect.png' 
   plt.savefig(pngpath, dpi=dpi)
   return pngpath
   
#plt.colorbar()
#plt.show()

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






