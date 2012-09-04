#!/usr/bin/python

import argparse
import math
import configobj

import sys
sys.path.append('/lab/software/apparatus3/py')

import statdat


def getval(r,seckey):
  tk = seckey.split(':') 
  return float( r[tk[0]][tk[1]])
 
def getstr(r,seckey):
  tk = seckey.split(':') 
  return r[tk[0]][tk[1]]

# --------------------- MAIN CODE  --------------------#

if __name__ == "__main__":
  parser = argparse.ArgumentParser('getevapimage.py')
  parser.add_argument('SHOT', action="store", type=int, help='shotnumber')
  parser.add_argument('CPOW', action="store", type=float, help='desired cpow') 

  args = parser.parse_args()
  #print type(args)
  #print args
  shotstr = "%04d" % args.SHOT
  report = "report" + shotstr + ".INI" 

  try:
    r = configobj.ConfigObj( report )
    ramp = getstr(r,'EVAP:ramp')
    ss = getval(r,'EVAP:evapss')
    finalcpow = getval(r,'EVAP:finalcpow')
    print finalcpow
  except:
    ramp = None
    print "Could not find ramp in report!" 

  if finalcpow > args.CPOW:
    print "This shot does not go as low as %.3f" % args.CPOW
    print "finalcpow = %.3f" % finalcpow

  else:
    ramp = ramp.replace('L:','/lab')
    rampdat = open( ramp, 'r')
    cpows = rampdat.readlines()
    #print cpows
    rampdat.close()
    image = 0
    while float(cpows[image]) > args.CPOW and image<len(cpows)-1:
      image = image+1
    image = image - 1
    print "cpow = %.3f at EVAP:image = %.2f" % ( float(cpows[image]), image*ss )

    
  exit(0)
