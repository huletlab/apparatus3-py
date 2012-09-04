#!/usr/bin/python

import sys
import getopt
import math

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
  
  U  = ramp*U0/10.
  
  h = 48./1000. # uK/kHz
  vr0 = 3.8     # kHz
  va0 = 3.8/8.  # kHz

  TFfactor = h * ((vr0**2 * va0)**(1./3.)) *  (U/U0)**(1./2.) # Multiply by (6*N)^1/3 to get T_F

  print U, TFfactor
  return

def verbose(nargs):
  print ("len(sys.argv) = %d" % nargs)
  print ("t\t= %.2f" % t)
  print ("free\t= %.2f" % free)
  print ("p1\t= %.2f" % p1)
  print ("t1\t= %.2f" % t1)
  print ("tau\t= %.2f" % tau)
  print ("beta\t= %.2f" % beta)
  print ("p0\t= %.2f" % p0)
  print ("offset\t= %.2f" % offset)
  print ("t2\t= %.2f" % t2)
  print ("tau2\t= %.2f" % tau2)
  print ("U0\t= %.2f" % U0)


if __name__ == '__main__':
  if len(sys.argv) != 12:
    print( "usage:  evapramp2.py  t  free  p1  t1  tau  beta  p0  offset  t2  tau2  U0") 
    exit(2) 

  t     = float(sys.argv[1])*1000.
  free  = float(sys.argv[2])
  p1    = float(sys.argv[3])
  t1    = float(sys.argv[4])
  tau   = float(sys.argv[5])
  beta  = float(sys.argv[6])
  p0    = float(sys.argv[7])
  offset= float(sys.argv[8])
  t2    = float(sys.argv[9])
  tau2  = float(sys.argv[10])
  U0    = float(sys.argv[11])
  
  #verbose(len(sys.argv))

  main()


	
	
	
