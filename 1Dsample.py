import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import qrange

import numpy as np

import abel
import argparse

from scipy.constants import *


def process(shots, **kwargs):
    imgs = []
    transformed = []
    shots = [int(shot) for shot in shots]
    for shot in shots:
        img = np.loadtxt("{0}_column.ascii".format(shot))
        img = np.rot90(img)

        img = abel.center_image(img, center="gaussian")
        imgs.append(img)
        tr = abel.Transform(img, method="hansenlaw", symmetry_axis=(None), symmetrize_method="fourier").transform
        transformed.append(tr)

    imgs.np.array(img)

    x = np.array(range(img.shape[1])) * 1.5
    y = np.array(range(img.shape[0])) * 1.5
    plt.figure(figsize=(5, 10))
    plt.subplot("211")
    plt.pcolor(x, y, np.mean(imgs, axis=0))
    plt.xlim(30, 120)
    plt.ylim(30, 120)
    plt.subplot("212")
    plt.errorbar(x, np.mean(transformed, axis=0).sum(axis=0) * (0.01 / (1.5 * micro)) ** 2 * (532 * nano / 0.01) ** 2,
                 yerr=np.sum(transformed, axis=1).std(axis=0) * (0.01 / (1.5 * micro)) ** 2 * (532 * nano / 0.01) ** 2)
    plt.xlim(30, 120)
    plt.ylabel("atoms per tube")
    plt.xlabel("um")
    plt.savefig("{0}-{1}_1D_sample.png".format(shots[0], shots[-1]))
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--range', action="store", dest='range', help="analyze files in the specified range.")
    args = parser.parse_args()
    datadir = "./"
    shots = qrange.parse_range(args.range)
    process(shots)
