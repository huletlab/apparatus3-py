#!/lab/software/epd-7.0-1-rh5-x86/bin/python

import sys
import getopt
import math

def main():
  t=float(sys.argv[1])*1000.
  free=float(sys.argv[2])
  p1=float(sys.argv[3])
  t1=float(sys.argv[4])
  tau=float(sys.argv[5])
  beta=float(sys.argv[6])
  p0=float(sys.argv[7])
  U0=float(sys.argv[8])
  
  if  t < free :
    ramp=p0
  elif t < free + t1:
    ramp=(p0-p1)*math.tanh( beta/tau * (t-t1-free) * p1 / (p0-p1) )/math.tanh( beta/tau * (-t1) * p1 / (p0-p1)) + p1
  else:
    ramp= p1*(1**beta)/(1 + (t-t1-free)/tau)**beta
  
  print ramp*U0/10.

  return


if __name__ == '__main__':
    main()


	
	
	
