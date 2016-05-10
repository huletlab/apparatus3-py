#!/usr/bin/python

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
sys.path.append('/lab/software/apparatus3/py')
import qrange, statdat, fitlibrary
from uncertainties import ufloat,unumpy
from configobj import ConfigObj

from scipy import stats

import argparse
parser = argparse.ArgumentParser('showbragg')
parser.add_argument('--range', action = "store", \
       help="range of shots to be used.")
parser.add_argument('--image', action="store", type=float, default=285.5,\
       help="value of image for in-situ bragg data")
parser.add_argument('--tofval', action="store", type=float, default=0.006,\
       help="value of DL.tof that represents a TOF shot")
parser.add_argument('--braggdet', action="store",type=float, default=-117.,\
       help="value of detuning for Bragg shots")
parser.add_argument('--latticedepth', action="store",type=float, default=5.5,\
       help="depth of hte lattice in Er")
parser.add_argument('--bin', action="store",type=float, default=500,\
       help="bin of andor 1 tof count")
parser.add_argument('--varyimage', action="store_true", default=False,\
       help="use this if image key is varying throught set")

aSkey = 'DIMPLELATTICE:knob05' 

parser.add_argument('--xkey', action="store", type=str, default=aSkey,\
       help="name of the report key for the X axis")


args = parser.parse_args()




import datetime
datestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_')

if args.xkey != aSkey:
    xkeystr = "_%s_"%args.xkey
    xkeystr = xkeystr.replace(':','_') 
else:
    xkeystr = '' 

gdat = {} 
gdat[ "udata_" + datestamp ] = {\
    'label':'udata',\
    'dir':os.getcwd(),\
    'shots':args.range,\
    'ec':'blue', 'fc':'blue',\
} 
  
datakeys = [args.xkey, 'SEQ:shot', 'ANDOR1EIGEN:signal',\
            'ANDOR2EIGEN:signal', 'HHHEIGEN:andor2norm',\
            'DIMPLELATTICE:force_lcr3', 'DIMPLELATTICE:tof',\
            'DIMPLELATTICE:imgdet', 'DIMPLELATTICE:image' ]

from DataHandling import data_fetch, data_ratio, data_pick, plotkey, plotkey_ratio 

gdat, K  = data_fetch( datakeys, gdat, save=False) 


def roundUpToMultiple(number, multiple):
    num = number + (multiple - 1)
    return num - (num % multiple)


for k in sorted(gdat.keys()):
    dat = gdat[k]['data']
    tofLock_cond = [ ('DIMPLELATTICE:tof', args.tofval),\
                     ('DIMPLELATTICE:imgdet', args.braggdet)]
    if not args.varyimage:
        tofLock_cond = tofLock_cond +  [('DIMPLELATTICE:image', args.imageTOFlock)]

    inSitu_cond = [('DIMPLELATTICE:tof',0.0),('DIMPLELATTICE:imgdet',args.braggdet),
                    ('DIMPLELATTICE:force_lcr3', -1) ]

    if not args.varyimage:
        inSitu_cond = inSitu_cond + [('DIMPLELATTICE:image',args.image)]

    tofLock  = data_pick( dat, tofLock_cond , K ) 
    inSitu   = data_pick( dat, inSitu_cond, K )
    print "LOCKTOF DATA  @", np.unique(tofLock[:,K(args.xkey)])
    print "INSITU DATA   @", np.unique(inSitu[:,K(args.xkey)])

    tofLockset = set( np.unique(tofLock[:,K(args.xkey)]).tolist() )
    inSituset = set( np.unique(inSitu[:,K(args.xkey)]).tolist() )
    common = list( tofLockset & inSituset ) 
    print common
    np.set_printoptions(suppress=True, precision=3)
    for i,c in enumerate(sorted(common)):
	tofLocki = tofLock[ tofLock[:,K(args.xkey)] == c ]
        inSitui = inSitu[ inSitu[:,K(args.xkey)] == c ]
	for n,j in enumerate(inSitui):
		index, shot = min(enumerate(tofLocki[:,K("SEQ:shot")]),key=lambda l: abs(l[1]-j[K("SEQ:shot")]))
		ishot = "report%04d.INI" %j[K('SEQ:shot')]
		tshot = "report%04d.INI" %shot
		ireport = ConfigObj(ishot)
		treport = ConfigObj(tshot)
		ireport["ANDOR1extra"]={}
		ireport["ANDOR2extra"]={}
		treport["ANDOR1extra"]={}
		treport["ANDOR2extra"]={}
		ireport["ANDOR1extra"]["nearestTof_signal"]=tofLocki[index,K('ANDOR1EIGEN:signal')]
		ireport["ANDOR1extra"]["nearestTof_shot"]=shot
		ireport["ANDOR2extra"]["nearestTof_signal"]=tofLocki[index,K('ANDOR2EIGEN:signal')]
		ireport["ANDOR2extra"]["nearestTof_shot"]=shot
		treport["ANDOR1extra"]["nearestTof_signal"]=tofLocki[index,K('ANDOR1EIGEN:signal')]
		treport["ANDOR1extra"]["nearestTof_shot"]=shot
		treport["ANDOR2extra"]["nearestTof_signal"]=tofLocki[index,K('ANDOR2EIGEN:signal')]
		treport["ANDOR2extra"]["nearestTof_shot"]=shot
		ireport.write()
		treport.write()

