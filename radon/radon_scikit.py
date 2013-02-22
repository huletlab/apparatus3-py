"""
===============
Radon transform
===============

The radon transform is a technique widely used in tomography to
reconstruct an object from different projections. A projection is, for
example, the scattering data obtained as the output of a tomographic
scan.

For more information see:

  - http://en.wikipedia.org/wiki/Radon_transform
  - http://www.clear.rice.edu/elec431/projects96/DSP/bpanalysis.html

This script performs the radon transform, and reconstructs the
input image based on the resulting sinogram.

"""

import matplotlib.pyplot as plt
import numpy
from skimage.io import imread
from skimage import data_dir
from skimage.transform import radon, iradon
from scipy.ndimage import zoom
import math


plt.figure(figsize=(9, 8.5), dpi=75)

plt.subplot(221)
def sphere(x,r):
	return(r**2-x**2)**0.5 if abs(x)<r else 0 #(r**2-x**2)**0.5
length = 1001
x =[ i - math.floor(length*0.5) for i in range(length)]
y= [ sphere(i,length*0.3) for i in x]
plt.plot(y);
plt.title("Original Cut")
plt.xlabel("Projection axis");
plt.ylabel("Intensity");


plt.subplot(222)

sino = numpy.array(y*100).reshape(100,length).transpose()
plt.title("Radon transform\n(Sinogram)");
plt.xlabel("Projection axis");
plt.ylabel("Intensity");
plt.imshow(sino)


reconstruction = iradon(sino)
plt.subplot(223)
plt.title("Cut Reconstruction\nfrom sinogram")
y= reconstruction[:,math.ceil(len(reconstruction)*0.5)]
plt.plot(y);
plt.title("Reconstruct Cut")
plt.xlabel("Projection axis");
plt.ylabel("Intensity");


plt.subplot(224)
plt.title("Reconstruction\nfrom sinogram")
plt.imshow(reconstruction, cmap=plt.cm.Greys_r)
plt.subplots_adjust(hspace=0.4, wspace=0.5)
plt.show()

