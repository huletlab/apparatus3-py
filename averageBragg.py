#!/usr/bin/python

import sys
import argparse
import os
import glob
from configobj import ConfigObj

import numpy as np
import matplotlib.pyplot as plt
import itertools

sys.path.append('/lab/software/apparatus3/py')

import falsecolor
import gaussfitter
import qrange

def rebin(a, binsz, rollX, rollY):
    a = np.roll(np.roll(a, rollX, axis=0), rollY, axis=1)
    shape = (a.shape[0]/binsz , a.shape[1]/binsz)
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    #print "Original shape : ", a.shape
    #print "     New shape : ", shape
    #print " Process shape : ", sh
    return a.reshape(sh).sum(-1).sum(1)

def steppify(arr,isX=False,interval=0):
    """
    Converts an array to double-length for step plotting
    """
    if isX and interval==0:
        interval = np.abs(arr[1]-arr[0]) / 2.0
        newarr = np.array(zip(arr-interval,arr+interval)).ravel()
        return newarr
    if not isX and interval==0:
        newarr = np.array(zip(arr,arr)).ravel()
        return newarr

def get_offset( DAT, box):
  col = DAT.max(axis=0).argmax()
  row = DAT.max(axis=1).argmax()
 
  br0 = 1+row+np.ceil(-box/2)
  br1 = 1+row+np.ceil(box/2)
  bc0 = 1+col+np.ceil(-box/2)
  bc1 = 1+col+np.ceil(box/2)

  boxDAT = DAT[ br0:br1 , bc0:bc1]
  #print boxDAT

  mask = np.ones( DAT.shape)
  mask[ br0:br1, bc0:bc1 ] = 0.

  offset = np.sum( mask * DAT ) \
           / ( DAT.shape[0]*DAT.shape[1] - boxDAT.shape[0]*boxDAT.shape[1] ) 

  return offset

# --------------------- MAIN CODE  --------------------#

if __name__ == "__main__":
  parser = argparse.ArgumentParser('averageBragg.py')

  parser.add_argument('range', action = "store", \
         help="average files in the specified range.")

  parser.add_argument('KEY', action="store", \
         type=str, help='using this key as discriminator')

  parser.add_argument('--binsz', \
         action="store", type=int, \
         help='binsz  bin size for rebinning')

  parser.add_argument('--roll', \
         action="store", type=str, \
         help='rollX, rollY  amount to roll matrix to get \
                             better binning')

  args = parser.parse_args()

  if not args.binsz:
      args.binsz = 4
 

  print "...Using binsize = %d" % args.binsz

  if not args.roll:
      rollX = 0 
      rollY = 0
  else:
      rollX = int(args.roll.split(',')[0])
      rollY = int(args.roll.split(',')[1])
 
  print "...Using rollX = %d" % rollX
  print "...Using rollY = %d" % rollY

  rangestr = args.range.replace(':','to')
  rangestr = rangestr.replace(',','_')
 
  #sec = args.KEY.split(':')[0]
  #key = args.KEY.split(':')[1]

  i = 0
  keydict = {}
  sqs  = {}
  sums = {}
  ns   = {}
  vals = {}
 
  sumnet6 = {}

  shots = qrange.parse_range(args.range)
  shlist = glob.glob( os.getcwd() + '/????_bragg.dat')
  shlist.sort()

  # Collect data for all shots, sums and Ns
  for sh in shlist:
    s =  os.path.split(sh)[1].split('_bragg')[0]
    if s in shots:
 
        report = ConfigObj( 'report'+s+'.INI')
        #val = report[sec][key]
        val = qrange.evalstr(report, args.KEY)
        
        dat = rebin(np.loadtxt( s + '_bragg.dat'), \
                    args.binsz, rollX, rollY) 

        if val in keydict.keys():
            index = keydict[val]
            sums[index] = sums[index] + dat
            sqs[index]  = sqs[index] + dat*dat
            ns[index] = ns[index] + 1

            
            sumnet6[index].append( float(report['BRAGG']['sum_net_6']))
 
        else:
            keydict[val] = i
            index =  i
            vals[index] = val 
            i = i + 1
            ns[index] = 1
            sums[index] = dat
	    sqs[index]  = dat*dat

            sumnet6[index] = [ float(report['BRAGG']['sum_net_6']) ]
            print type(sumnet6[index])


  # Obtain the averages
  means={}
  sqmeans={}
  stddevs={}
  cmin = 1e6
  cmax = -1e6
  for index in range(i):
    means[index] = sums[index] / ns[index]
    sqmeans[index] = sqs[index] / ns[index]
    stddevs[index] = np.sqrt( sqmeans[index] - means[index] ) 

    means[index] = means[index] - get_offset( means[index] , 3) 

    if means[index].max() > cmax:
        cmax = means[index].max()
    if means[index].min() < cmin:
        cmin = means[index].min()

  fig = plt.figure()

  kwargs = {}
  kwargs['vmin'] = cmin
  kwargs['vmax'] = cmax
  kwargs['cmap'] = falsecolor.my_grayscale
  kwargs['interpolation'] = 'nearest' 
  kwargs['origin'] = 'lower' 

  for a in range(i):
     ax = fig.add_subplot(i+1, i+1,  a+2)
     ax.imshow( means[a], **kwargs)
     ax = fig.add_subplot(i+1, i+1,  (i+1)*(a+1)+1 )
     ax.imshow( means[a], **kwargs)

  kwargs = {}
  kwargs['cmap'] = falsecolor.my_grayscale
  kwargs['interpolation'] = 'nearest' 
  kwargs['origin'] = 'lower' 

  for a,b in itertools.combinations( range(i), 2):
     diff = means[b] - means[a]
     row, col = np.unravel_index( diff.argmax(), diff.shape) 
     
     ax = fig.add_subplot(i+1, i+1,  (i+1)*(a+1)+(b+2) )
     #ax.imshow(diff, **kwargs)
     #ax2 = ax.twinx()
     x = np.arange(means[b][row,:].size)

     print "%d : %.2f +/- %.2f" % (b, means[b][row,col], stddevs[b][row,col])
     print "%d : %.2f +/- %.2f" % (a, means[a][row,col], stddevs[a][row,col])
    
     lB = means[b][row,:]
     lA = means[a][row,:]
     
     xx = steppify(x, isX=True)
 
     sB = stddevs[b][row,:]
     sA = stddevs[a][row,:]
    
     sBminus = steppify( lB-sB )
     sBplus  = steppify( lB+sB ) 
   
     sAminus = steppify( lA-sA )
     sAplus  = steppify( lA+sA ) 
     
   
     ax.step(x, lB-sB, 'b',where='mid', alpha=0.5)
     ax.step(x, lB+sB, 'b',where='mid', alpha=0.5)
     ax.step(x, lA-sA, 'g',where='mid', alpha=0.5)
     ax.step(x, lA+sA, 'g',where='mid', alpha=0.5)
  
     ax.fill_between(xx, sBplus, sBminus, facecolor='b',alpha=0.4)
     ax.fill_between(xx, sAplus, sAminus, facecolor='g',alpha=0.4)
  
     ax.step(x, lB,'b', where='mid', lw=2.)
     ax.step(x, lA,'g', where='mid', lw=2.)
    
     ax.set_xlim( x.min(), x.max())
    
 
     ax = fig.add_subplot(i+1, i+1,  (i+1)*(b+1)+(a+2) )
     ax.imshow( diff, **kwargs)

  plt.show()

  

