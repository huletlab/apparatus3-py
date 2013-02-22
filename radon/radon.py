# Inversion of 3D densities using the Gaussian basis-set expansion

import numpy as np

def rho(k,s,r):
  print k
  if k == 0:
    return   np.power( r / s, 2 * k**2) * np.exp( - (r/s)**2)
  else:
    return   np.power( np.exp(1) / k**2 , k**2) * np.power( r / s, 2 * k**2) * np.exp( - (r/s)**2)

R = 10.
r = np.arange(-R,R, 0.01	) 

from matplotlib import pyplot

for k in range(3):
  pyplot.plot( r, rho(k,1,r), '-') 

pyplot.show()


