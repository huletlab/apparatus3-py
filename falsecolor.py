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

def inspecpng( imglist , inspec_row, inspec_col, cmin, cmax, colormap, id, dpi, origin='lower', scale=1.0,step=False, interpolation=None, extratext='',alphadata=0.35, alphacross=0.4):
   """Makes an inspector plot (falsecolor plus cuts) for the list of images, the first image
      is always the data and the rest are treated as fits"""
   shape = imglist[0].shape
   for img in imglist:
     if img.shape != shape:
       print " --->  ERROR:  falsecolor.inspecpng  cannot handle images of different shapes"
       exit(1)

   #Inspec png accepts float for row and col
   finspec_row = float(inspec_row)
   finspec_col = float(inspec_col)
   
   inspec_row = int(np.floor(float(inspec_row)))
   inspec_col = int(np.floor(float(inspec_col)))

   #grid pixels per inch
   gpxi = 75. / scale

   #size of main axes in inches
   axw_in = shape[1]/gpxi
   axh_in = shape[0]/gpxi
  
   #scale of fit axes 
   fits = 0.4

   #scale of plots area
   plotw_in = axw_in*(1. + fits)
   ploth_in = axh_in*(1. + fits)

   #size offigure
   xlabel_border = 0.7 # inches
   ylabel_border = 0.7 # inches
   gap = 0.2 # inches 
   figw_in = xlabel_border + plotw_in + 2*gap
   figh_in = ylabel_border + ploth_in + 2*gap

   
   fig = plt.figure(figsize=(figw_in,figh_in))

   figuresize = [figw_in, figh_in]


   #
   xstart = xlabel_border/figw_in
   ystart = ylabel_border/figh_in
   xsize  = axw_in/figw_in
   ysize  = axh_in/figh_in
   axRect = [xstart, ystart, xsize, ysize]
   ax = fig.add_axes(axRect,frameon=True)
   im = ax.imshow(imglist[0], cmap=colormap, vmin=cmin,vmax=cmax, interpolation=interpolation)

   centertext = 'shot #%s\ncenter at (%.1f, %.1f)' % (id,finspec_col,finspec_row)
   centertext = centertext + extratext
   fig.text(xstart-0.65/figw_in, ystart-0.65/figh_in, centertext)

   #
   xstart = (xlabel_border + gap + axw_in)/figw_in 
   ystart = (ylabel_border + gap + axh_in)/figh_in 
   xsize  = fits*axw_in/figw_in
   ysize  = fits*axh_in/figh_in
   axFITRect = [xstart, ystart, xsize, ysize]
   axFIT = fig.add_axes( axFITRect, frameon=True)
   if len(imglist) > 1 :
     axFIT.imshow( imglist[1], cmap=colormap, vmin=cmin, vmax=cmax, interpolation=interpolation)
   axFIT.xaxis.set_ticklabels([])
   axFIT.yaxis.set_ticklabels([])
 
  
   #THIS IS THE LINE PLOT ON THE TOP 
   ax.axhline( finspec_row, linewidth=0.8, color='black', alpha=alphacross)
   #
   ###axROW = fig.add_axes([xstart,ygap+axh, axw, ys], frameon=True)
   xstart = xlabel_border/figw_in
   ystart = (ylabel_border + gap + axh_in)/figh_in 
   xsize  = axw_in/figw_in
   ysize  = fits*axh_in/figh_in
   axROWRect = [xstart, ystart, xsize, ysize] 
   axROW = fig.add_axes(axROWRect, frameon=True)
   axROW.set_xlim( 0, len(imglist[0][ inspec_row, :])-1)
   if step:
     axROW.step( np.arange(0, len(imglist[0][inspec_row,:]), 1),  imglist[0][ inspec_row, :] , color='blue', alpha=alphadata, where='mid')
   else:
     axROW.plot( imglist[0][ inspec_row, :] , color='blue', alpha=alphadata)
   for img in imglist[1:]:
     axROW.plot( img[ inspec_row, :] , color='blue')
   axROW.xaxis.set_ticklabels([])
 

   #THIS IS THE LINE PLOT ON THE RIGHT
   ax.axvline( finspec_col, linewidth=0.8, color='black', alpha=alphacross)
   #
   ###axCOL = fig.add_axes([xgap+axw,ystart, xs , axh], frameon=True)
   xstart = (xlabel_border + gap + axw_in)/figw_in 
   ystart = ylabel_border/figh_in
   xsize  = fits*axw_in/figw_in
   ysize  = axh_in/figh_in
   axCOLRect = [xstart, ystart, xsize, ysize]
   axCOL = fig.add_axes(axCOLRect, frameon=True)

   xarray = np.arange( len( imglist[0][:, inspec_col]) )
   if origin == 'lower':
     axCOL.set_ylim( 0, len( imglist[0][ :, inspec_col])-1)
   else:
     axCOL.set_ylim( len( imglist[0][ :, inspec_col])-1, 0)
   if step:
     xarray = xarray + 0.5
     axCOL.step( imglist[0][ : , inspec_col], xarray , color='red', alpha=alphadata, where='mid')
   else:
     axCOL.plot( imglist[0][ : , inspec_col], xarray , color='red', alpha=alphadata)
   for img in imglist[1:]:
     axCOL.plot( img[ : , inspec_col], xarray , color='red')
   axCOL.yaxis.set_ticklabels([]) 
   labels = axCOL.get_xticklabels()
   for label in labels:
     label.set_rotation(-90)

   ax.set_xlim ( 0, len(imglist[0][ inspec_row, :])-1 )
   
   if origin == 'lower': 
     ax.set_ylim ( 0, len(imglist[0][ :, inspec_col])-1)
     axFIT.set_ylim ( 0, len(imglist[0][ :, inspec_col])-1)
   
   labels = ax.get_xticklabels()
   for label in labels:
     label.set_rotation(-90)

   pngpath = id + '_inspect.png' 
   plt.savefig(pngpath, dpi=dpi)

  
   axesdef = [ axRect, axROWRect, axCOLRect, axFITRect]

   return pngpath, figuresize, axesdef
   
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






