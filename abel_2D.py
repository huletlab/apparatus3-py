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

import os

from scipy.ndimage import rotate

'''
def glvd(img):
    assert isinstance(img, np.ndarray)
    return np.sum((img[:, ::-1] - img) ** 2)
'''

factor = (532*nano)**2/(1.5*micro)**2

def abel_2d(shots, fname=None, n_iter=None, center="gaussian"):
    imgs = []
    for shot in shots:
        img = np.loadtxt("{0}_column.ascii".format(shot))
        img = np.rot90(img)
        img = abel.center_image(img, center=center)
        imgs.append(img)

    imgs = np.array(imgs)

    if n_iter is None:
        n_iter = len(shots)
    N = len(shots)
    imgs_transformed = []

    for i in range(n_iter):
        img_av = imgs[np.random.randint(0, N, N)].mean(axis=0)
        tr = abel.Transform(img_av, method="hansenlaw", symmetry_axis=(None), symmetrize_method="fourier",
                            center=center).transform
        imgs_transformed.append(tr)

    tr_av = np.mean(imgs_transformed, axis=0)*factor
    tr_std = np.std(imgs_transformed, axis=0)*factor

    x0 = tr_av.shape[0] / 2 + tr_av.shape[0] % 2
    x1 = tr_av.shape[1] / 2 + tr_av.shape[1] % 2
    plt.figure(figsize=(10, 10))

    plt.subplot("221")
    plt.pcolormesh(tr_av)
    plt.colorbar()

    plt.subplot("222")
    plt.errorbar( tr_av.T[x1], range(tr_av.shape[0]), xerr=tr_std.T[x1])
    plt.xlabel("# per lattice site")

    plt.subplot("223")
    plt.errorbar(range(tr_av.shape[1]), tr_av[x0], yerr=tr_std[x0])
    plt.ylabel("# per lattice site")

    plt.subplot("224")
    plt.pcolormesh(np.mean(imgs, axis=0))
    plt.colorbar()
    plt.savefig(fname)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--range', action="store", dest='range', help="analyze files in the specified range.")
    args = parser.parse_args()

    shots = qrange.parse_range(args.range)
    if not os.path.isdir("plots"):
        os.mkdir("plots")
    fname = os.path.join("plots", "{0}_2D_abel.png".format(args.range))
    abel_2d(shots, fname)
