#!/usr/bin/python
import matplotlib

try:
    matplotlib.use("Agg")
except:
    pass

import matplotlib.pyplot as plt

try:
    plt.style.use('ggplot')
except:
    pass
import qrange

import numpy as np

import abel
import argparse

from scipy.constants import *
from scipy.fftpack import fft2, ifft2, fftfreq

from scipy.misc import imresize

import os

factor = (532 * nano / (1.497 * micro)) ** 2
factor_density = (532 * nano) ** 3 / (1.497 * micro) ** 3



def butter(img, k0, n, d=1.0):
    """
    Butterworth filter
    """
    kimg = fft2(img)

    n0, n1 = img.shape

    kk0, kk1 = np.meshgrid(fftfreq(n1, d=d), fftfreq(n0, d=d))

    k = np.sqrt(kk0 ** 2 + kk1 ** 2)

    B = 1.0 / (1 + (k / k0) ** (2 * n))

    kimg = kimg * B

    return ifft2(kimg).real

def process(shots, fname, n_iter=None, save_txt=False, folder='plots', resize_fac=1.0,**kwargs):
    imgs = []
    transformed = []
    shots = [int(shot) for shot in shots]
    for shot in shots:
        img = np.loadtxt("{0:04d}_column.ascii".format(int(shot)))
        img = np.rot90(img)
        #

        img = abel.center_image(img, center="gaussian")
        img = butter(img, 0.5, 10)
        img = img[15:-15, 7:-7]

#        print img
        #img = imresize(img, resize_fac, interp="cubic")
        imgs.append(img)

    imgs = np.array(imgs)
    img_ave = imgs.mean(axis=0)
    y1D_original = np.sum(img_ave, axis=1)

    if n_iter is None:
        n_iter = len(shots)
    N = len(shots)

    for i in range(n_iter):
        img_av = imgs[np.random.randint(0, N, N)].mean(axis=0)
        tr = abel.Transform(img_av, method="three_point", symmetry_axis=(0), symmetrize_method="fourier").transform
        #tr = butter(tr, 0.2, 10)
        transformed.append(tr)

    tr_av = np.mean(transformed, axis=0) * factor_density
    tr_std = np.std(transformed, axis=0) * factor_density
    img_av = np.mean(imgs, axis=0)
    img_std = np.std(imgs, axis=0)
    y1D_av = np.mean(transformed, axis=0).sum(axis=0) * factor
    y1D_std = np.sum(transformed, axis=1).std(axis=0) * factor

    x = np.array(range(img.shape[1])) * 1.497 / resize_fac
    y = np.array(range(img.shape[0])) * 1.497 / resize_fac

    plt.figure(figsize=(5, 15))
    plt.subplot("311")
    plt.pcolor(x, y, img_av)
    plt.xlim(0, x.max())
    plt.ylim(0, y.max())
    plt.xlabel("um")
    plt.ylabel("um")
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(r"number per pixel")
    plt.title("column density")

#    plt.subplot("412")
#    plt.plot(y1D_original)
#    #plt.xlim(0, x.max())
#    plt.ylabel("atoms per pixel")
#    plt.xlabel("um")

    plt.subplot("312")
    plt.pcolor(x, y, tr_av)
    plt.xlim(0, x.max())
    plt.ylim(0, y.max())
    plt.xlabel("um")
    plt.ylabel("um")
    cbar2 = plt.colorbar()
    cbar2.ax.set_ylabel(r"number per lattice site")
    plt.title("inverse Abel transform")

    plt.subplot("313")
    plt.errorbar(x, y1D_av, y1D_std)
    plt.xlim(0, x.max())
    plt.ylabel("atoms per tube")
    plt.xlabel("um")

    if not os.path.isdir(folder):
        os.mkdir(folder)

    fname = os.path.join(folder, fname)
    print "create", fname

    plt.savefig(fname)
    plt.close()

    if save_txt:
        np.savetxt(fname[:-4] + ".column_density_mean.txt", img_av)
        np.savetxt(fname[:-4] + ".column_density_std.txt", img_std)
        np.savetxt(fname[:-4] + ".inverse_abel_mean.txt", tr_av)
        np.savetxt(fname[:-4] + ".inverse_abel_std.txt", tr_std)
        np.savetxt(fname[:-4] + ".1d_profile_mean.txt", y1D_av)
        np.savetxt(fname[:-4] + ".1d_profile_std.txt", y1D_std)
        np.savetxt(fname[:-4] + ".shots.txt", shots)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--range', action="store", dest='range', help="analyze files in the specified range.")
    parser.add_argument('--fname', action="store", dest='fname', help="Filename",default= "")
    parser.add_argument('--folder', action="store", dest='folder', help="Folder",default= "plots")
    args = parser.parse_args()
    datadir = "./"
    fname = "{0}_1D_sample.png".format(args.fname or args.range)
    shots = qrange.parse_range(args.range)
    process(shots, fname=fname,folder = args.folder,save_txt=True)
