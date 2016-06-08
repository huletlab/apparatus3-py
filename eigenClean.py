#!/usr/bin/python
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

try:
    plt.style.use('ggplot')
except:
    pass

import numpy as np
import pyfits
import qrange
import argparse
import os
import shutil

from scipy import linalg

import cPickle

import warnings

warnings.filterwarnings("ignore")


class EigenEngine:
    def __init__(self):
        self.image_shape = None
        self.n = None
        self.S = None
        self.V = None
        self.av = None
        self.ROI = (None, None, None, None)
        self.scale_factor = None

    def calculate_scale_factor(self):
        # mask = np.zeros(shape=self.image_shape, dtype=np.bool)
        x0_start, x0_end, x1_start, x1_end = self.ROI
        # mask[x0_start:x0_end, x1_start:x1_end] = True
        # mask = mask.flatten()
        # self.scale_factor = 1.0/(1-np.array([np.sum(vec[mask]**2) for vec in self.V]))
        # self.scale_factor = 1.0/(1-np.array([linalg.norm(vec[mask]) for vec in self.V]))
        A = self.image_shape[0] * self.image_shape[1]
        self.scale_factor = A / (A - float((x0_end - x0_start) * (x1_end - x1_start)))

    def fit(self, data):
        self.image_shape = data[0].shape
        self.n = len(data)
        batch = np.zeros((self.n, self.image_shape[0] * self.image_shape[1]))
        for i in range(self.n):
            batch[i] = data[i].flatten()
        self.av = np.average(batch, axis=0)
        batch = batch - self.av
        U, self.S, self.V = linalg.svd(batch, full_matrices=0)
        self.scale_factor = None

    def predict_coeff(self, img, x0_start, x0_end, x1_start, x1_end, n=None):

        if self.ROI != (x0_start, x0_end, x1_start, x1_end):
            self.ROI = (x0_start, x0_end, x1_start, x1_end)
            self.scale_factor = None

        if self.image_shape != img.shape:
            pass
        n0, n1 = img.shape
        img_cp = img.copy()
        img_cp -= self.av.reshape(self.image_shape)
        img_cp[x0_start:x0_end, x1_start:x1_end] = 0
        img_cp = img_cp.flatten()

        if self.scale_factor is None:
            self.calculate_scale_factor()

        c = np.dot(self.V[:n], img_cp.T)
        if isinstance(self.scale_factor, np.ndarray):
            c *= self.scale_factor[:n]
        else:
            c *= self.scale_factor
        return c

    def predict(self, img, x0_start, x0_end, x1_start, x1_end, n=None):
        c = self.predict_coeff(img, x0_start, x0_end, x1_start, x1_end, n)
        return (np.dot(c, self.V[:n]) + self.av).reshape(self.image_shape)

    def save(self, fname):
        with open(fname, 'wb') as f:
            cPickle.dump(self.__dict__, f, 2)

    def load(self, fname):
        with open(fname, 'rb') as f:
            tmp_dict = cPickle.load(f)
        self.__dict__.update(tmp_dict)

    def remember_only(self, n):
        self.V = self.V[n]
        self.S = self.S[n]


class EigenClean:
    def __init__(self):
        self.EE = EigenEngine()

    def load(self, basis="basis"):
        self.EE.load(basis)

    @staticmethod
    def backup(shots):
        for shot in shots:
            try:
                src = "{0}noatoms.fits".format(shot)
                dst = "{0}noatoms.fits.backup".format(shot)
                if os.path.isfile(dst):
                    continue
                shutil.copy(src, dst)
            except Exception as err:
                print err

    @staticmethod
    def recover(shots):
        for shot in shots:
            try:
                src = "{0}noatoms.fits.backup".format(shot)
                dst = "{0}noatoms.fits".format(shot)
                shutil.copy(src, dst)
            except Exception as err:
                print err

    def learn(self, shots, basis=None, plot=False, n=None):
        imgs = []
        for shot in shots:
            fname = "{0}noatoms.fits.backup".format(shot)
            if not os.path.isfile(fname):
                fname = "{0}noatoms.fits".format(shot)

            with pyfits.open(fname) as p:
                img = p[0].data[0]

            imgs.append(img)

        self.EE.fit(imgs)

        if basis is None:
            basis = "basis"

        if n is not None:
            self.EE.remember_only(n)

        self.EE.save(basis)

        if plot:
            plt.plot(self.EE.S)
            plt.savefig("report_learn.png")
            plt.close()

    def predict(self, shots, roi, basis="basis", plot=False, verbose=True, n=None):
        if verbose:
            print "Running backup"
        self.backup(shots)
        roi_converted = roi_convert(roi)

        if verbose:
            print "Loading basis"

        if self.EE.av is None:  # No basis are loaded.
            self.EE.load(basis)

        for shot in shots:
            if verbose:
                print "Predicting #{0}".format(shot)
            atom = "{0}atoms.fits".format(shot)
            noatom = "{0}noatoms.fits".format(shot)
            bak = "{0}noatoms.fits.backup".format(shot)
            with pyfits.open(atom) as p_atom:
                with pyfits.open(noatom) as p_noatom:
                    img_atom = p_atom[0].data[0]

                    # print roi_converted
                    bg_predict = self.EE.predict(img_atom, *roi_converted, n=n)
                    bg_predict = bg_predict.astype(img_atom.dtype)
                    p_noatom[0].data[0] = bg_predict
                    # p_noatom.flush()
                    p_noatom.writeto(noatom, clobber=True)

                    if plot:
                        with pyfits.open(bak) as p_bak:
                            bg_original = p_bak[0].data[0]
                        x = np.arange(0, img_atom.shape[1])
                        y = np.arange(0, img_atom.shape[0])
                        od_original = np.divide(img_atom, bg_original, dtype=np.float)
                        od_new = np.divide(img_atom, bg_predict, dtype=np.float)
                        od_original_cp = od_original.copy()
                        od_new_cp = od_new.copy()
                        od_original_cp[roi_converted[0]:roi_converted[1], roi_converted[2]:roi_converted[3]] = 1
                        od_new_cp[roi_converted[0]:roi_converted[1], roi_converted[2]:roi_converted[3]] = 1

                        plt.figure(figsize=(10, 10))

                        plt.subplot("221")
                        plt.title("Original")

                        plt.pcolormesh(od_original)
                        plt.colorbar()
                        plt.xlim(0, img_atom.shape[1])
                        plt.ylim(0, img_atom.shape[0])

                        plt.subplot("222")
                        plt.title("Optimized")
                        plt.pcolormesh(x, y, od_new)
                        plt.xlim(0, img_atom.shape[1])
                        plt.ylim(0, img_atom.shape[0])
                        plt.colorbar()

                        plt.subplot("223")
                        plt.title("Original BG")
                        plt.pcolormesh(od_original_cp)
                        plt.colorbar()
                        plt.xlim(0, img_atom.shape[1])
                        plt.ylim(0, img_atom.shape[0])

                        plt.subplot("224")
                        plt.title("Optimized BG")
                        plt.pcolormesh(x, y, od_new_cp)

                        plt.xlim(0, img_atom.shape[1])
                        plt.ylim(0, img_atom.shape[0])
                        plt.colorbar()

                        plt.savefig("{0}_prediction.png".format(shot))
                        plt.close()


def roi_convert(roi):
    x0_start = roi[1]
    x0_end = roi[1] + roi[3]
    x1_start = roi[0]
    x1_end = roi[0] + roi[2]
    return x0_start, x0_end, x1_start, x1_end


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--range', action="store", dest='range', help="")
    parser.add_argument('--ROI', action="store", dest='roi', help="")
    parser.add_argument('--backup', action='store_true')
    parser.add_argument('--recover', action="store_true")
    parser.add_argument('--learn', action='store_true')
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--no_plot', action='store_true')
    parser.add_argument('-n', action="store", dest='n', type=int, default=None)
    args = parser.parse_args()
    shots = qrange.parse_range(args.range)
    shots = [int(shot) for shot in shots]
    n = args.n

    if args.no_plot:
        plot = False
    else:
        plot = True

    EC = EigenClean()

    if args.backup:
        EC.backup(shots)
    elif args.recover:
        EC.recover(shots)
    elif args.learn:
        EC.learn(shots, plot=plot, n=n)
    elif args.predict:
        roi = args.roi.split(",")
        roi = [int(i) for i in roi]
        EC.predict(shots, roi, plot=plot, n=n)
