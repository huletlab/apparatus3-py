#!/usr/bin/python

import argparse
import math
import configobj

import sys
sys.path.append('/lab/software/apparatus3/py')

import statdat

# --------------------- MAIN CODE  --------------------#

def getval(r,seckey):
  tk = seckey.split(':') 
  return float( r[tk[0]][tk[1]]) 

def knee(t):
  return (p0-p1) * math.tanh( beta/tau * (t-free -t1) * p1 / (p0-p1) ) / math.tanh( beta/tau * (-t1) * p1 / (p0-p1) ) + p1 

def evap(t): 
  return p1 * (1**beta) / ( 1 + (t-free - t1)/tau )**beta

def evap2(t):
  return ( (1-offset) * p1 * (1**beta) / (1 + (t2-t1)/tau )**beta +  10 * offset ) / ( 1 + (t-free - t2)/tau2 )
   

def main():
  if  t < free :
    ramp = p0
  elif t < free + t1:
    ramp = (1-offset) * knee(t) + 10 * offset
  elif t < free + t2:
    ramp = (1-offset) * evap(t) + 10 * offset
  else:
    ramp = evap2(t) 
  
  #print "\tramp = %.3f" % ramp 
 
  U  = ramp*U0/10.
  
  h = 48./1000. # uK/kHz
  vr0 = 3.8     # kHz
  va0 = 3.8/8.  # kHz

  TFfactor = h * ((vr0**2 * va0)**(1./3.)) *  (U/U0)**(1./2.) # Multiply by (6*N)^1/3 to get T_F

  #print U, TFfactor
  return U, TFfactor

if __name__ == "__main__":
  parser = argparse.ArgumentParser('plotbeamprofiles.py')
  parser.add_argument('reports', nargs='*', help='list of dat files to fit')

  args = parser.parse_args()
  #print type(args)
  #print args

  for i,report in enumerate(args.reports):
    print "Processing %s ..." % report
    r = configobj.ConfigObj( report )
    
    p0 = getval(r,'ODT:odtpow')
    free = getval(r,'EVAP:free')
    image = getval(r,'EVAP:image')
    p1 = getval(r,'EVAP:p1')
    t1 = getval(r,'EVAP:t1')
    tau = getval(r,'EVAP:tau')
    beta = getval(r,'EVAP:beta')
    offset = getval(r,'EVAP:offset')
    t2 = getval(r,'EVAP:t2')
    tau2 = getval(r,'EVAP:tau2')
    t = (free + image)

#    print "\tfree = %.3f" % free
    print "\timage = %.3f" % image
#    print "\tp0 = %.3f" % p0
#    print "\tp1 = %.3f" % p1
#    print "\tt1 = %.3f" % t1
#    print "\ttau = %.3f" % tau
#    print "\tbeta = %.3f" % beta
#    print "\tt2 = %.3f" % t2
#    print "\ttau2 = %.3f" % tau2
#    print "\toffset = %.3f" % offset
    
    U0 = 280.
 
    U, TFfactor = main() 
 
    print "\tU     = %.3f" % U

    r['ODTCALIB']['maxpow'] = 29.0
    r['ODTCALIB']['maxdepth'] = U0
    r['ODTCALIB']['v0radial'] = 3800.
    r['ODTCALIB']['v0axial'] = 475.
    r['EVAP']['finalcpow'] = 10.*U/U0
    r['EVAP']['ramp'] = 'L:/software/apparatus3/ramps/Evap4_3eb39df0ca02d6582d31e4e1d529681a_phys'

    r.write()
    
    


  exit(1)
