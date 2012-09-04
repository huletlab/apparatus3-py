#!/usr/bin/python

import numpy as np
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import sys
sys.path.append('/lab/software/apparatus3/bin/py')
import shot2fcolor

from configobj import ConfigObj

import argparse


if __name__ == "__main__":
  parser = argparse.ArgumentParser('shot2false.py')
 
  parser.add_argument('SHOT', action="store", type=int, help='shot number')
  parser.add_argument('INDEX', action="store", type=int, help='center index of profile plot')
  parser.add_argument('WIDTH', action="store", type=float, help='number of rows/columns to be averaged')
  parser.add_argument('--xy', action="store", dest='XY', type=str, help='0 is profile along X, 1 is profile along Y')
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
  else:
    X0 = 0.
    Y0 = 0.
    XW = column.shape[0]
    YW = column.shape[1] 

  if args.XY=='1':
    XYflag = '_Y_'
    profil = column[ : , args.INDEX+np.floor(-args.WIDTH/2):args.INDEX+np.floor(args.WIDTH/2) ]
    column = column[ : , args.INDEX+np.floor(-20*args.WIDTH/2):args.INDEX+np.floor(20*args.WIDTH/2) ]
    if args.ROI:
      profil = profil[ Y0:Y0+YW, :]
      column = column[ Y0:Y0+YW, :]  
    data = np.sum(profil, axis=1)
  else:
    XYflag = '_X_'
    profil = column[ args.INDEX+np.floor(-args.WIDTH/2):args.INDEX+np.floor(args.WIDTH/2) , : ]
    column = column[ args.INDEX+np.floor(-20*args.WIDTH/2):args.INDEX+np.floor(20*args.WIDTH/2) , : ]
    if args.ROI:
      profil = profil[ :, X0:X0+XW]
      column = column[ :, X0:X0+XW]
    data = np.sum(profil, axis=0)
 
  print profil.shape 
  print data.shape
    
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
 
  partgray = shot2fcolor.part_cmap( cm.gray, 0.0, 0.8)
  invertgray = shot2fcolor.invert_cmap( partgray )
  partspectral = shot2fcolor.part_cmap( cm.spectral, 0.2, 0.95)  

  merged=shot2fcolor.merge_cmaps( shot2fcolor.part_cmap( cm.gray, 0.2, 0.75) , shot2fcolor.part_cmap(cm.spectral, 0., 0.5) , (cutoff-cmin)/(cmax-cmin) )

#merged=merge_cmaps( my_grayscale, my_rainbow, (cutoff-cmin)/(cmax-cmin) )
#merged=merge_cmaps( cm.gray, cm.spectral, (cutoff-cmin)/(cmax-cmin) )


  # Make auxiliary false color plot
  fig = plt.figure(figsize=(YW/ppi,XW/ppi))
  ax = fig.add_axes([0.0,0.04,1.0,0.92],frameon=True)

  ax.set_yticklabels([])
  ax.set_xticklabels([])

  im = ax.imshow(column, cmap=partspectral, vmin=cmin,vmax=cmax, origin='lower')
  ax = im.get_axes()

  # Make profile plot
  AR = 0.3 
  if args.XY == '1':
    fig2 = plt.figure(figsize=(YW/ppi, YW/ppi*AR))
  else:
    fig2 = plt.figure(figsize=(XW/ppi, XW/ppi*AR))
  
  ax2 = fig2.add_axes([0.05,0.05,0.9,0.9], frameon=True)
  ax2.set_yticklabels([])
  ax2.set_xticklabels([])
  
  if args.XY == '1':
    #ax2.plot(data,color='black')
    ax2.plot(data, '.', color='darkblue', markersize=2)
  else:
    #ax2.plot(data,color='black')
    ax2.plot(data, '.',color='darkgreen', markersize=2, markeredgewidth=0.5)
  
  
  if args.DIR:
     pngauxpath = args.DIR + shotnum + XYflag + '_shot2profile_aux.png' 
     pngpath = args.DIR + shotnum + XYflag + '_shot2profile.png'
     txtpath = args.DIR + shotnum + XYflag + '_shot2profile.txt' 
  else:
     pngauxpath = shotnum + XYflag +  '_shot2profile_aux.png' 
     pngpath = shotnum + XYflag + '_shot2profile.png' 
     txtpath = shotnum + XYflag + '_shot2profile.txt' 
     
  fig.savefig(pngauxpath, dpi=dpi)
  fig2.savefig(pngpath,dpi=dpi)
  np.savetxt(txtpath, profil)
   

  exit(1) 


# --------------------  START OF PLOTTING PART --------------------#

#for i in dir(ax):
#    print i

mpl.rc('mathtext',default='regular')

#Scale label
#ax.text(10,540,r"$100\ \mu\mathrm{m}$",fontsize=16)

#Paramters label 
sec = sys.argv[SECKEY].split(':')[0]
key = sys.argv[SECKEY].split(':')[1]
val = report[sec][key]
labeltext="%s:%s=%s" % (sec,key,val)
ax.text(10,-2,labeltext, fontsize=8)

#Other labels
labeltext="%s:%s=%.2e" % ('CPP','nfit',float(report['CPP']['nfit']))
ax.text(10, -9, labeltext, fontsize=8)

labeltext="%s:%s=%.2e" % ('ANDOR','phcdet',float(report['ANDOR']['phcdet']))
ax.text(10, -16, labeltext, fontsize=8)

labeltext="%s:%s=%.2e" % ('ODT','odttof',float(report['ODT']['odttof']))
ax.text(10, -23, labeltext, fontsize=8)


scalebar = patches.Rectangle((10,500),31.25,6,ec='black', fc='black')
ax.add_patch(scalebar)

#for i in dir(plt):
#    print i   

#plt.colorbar()
#plt.show()



#parameter indeces
SHOT=1
AX0S=2
AX1S=3
AX0W=4
AX1W=5
CMIN=6
CMAX=7
CXOVER=8
SECKEY=9
DIR=10


if not (len(sys.argv) == DIR or len(sys.argv) == DIR+1):
    print "  column.py:"
    print ""
    print "  Looks at the columndensity for a given [SHOT] and creates a"
    print "  false color image.  The colormap goes from [MIN] to [MAX] "
    print "  and has a crossover from black and white to rainbow at [XOVER]."
    print ""
    print "  The image is saved to a png file with the value of [SEC:KEY]"
    print "  preppending the name.  This way a lot of images can be indexed"
    print "  for example by time of flight."
    print ""
    print "  The image is saved in the directory [DIR].  If none is given"
    print "  it is saved in the current directory."

    print "  usage:  column.py [SHOT] [AX0S] [AX1S] [AX0W] [AX1W] [CMIN] [CMAX] [CXOVER] [SEC:KEY] [DIR]"  
    exit(1)





