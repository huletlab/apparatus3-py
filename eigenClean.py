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
    available_predict_method = ("scale_area", "least_square")

    def __init__(self):
        self.image_shape = None
        self.N = None
        self.S = None
        self.V = None
        self.av = None
        self.ROI = (None, None, None, None)
        self.predict_method = None
        self.n = None
        # cache parameter
        self.mask = None
        self.predict_factor = None

    def remove_predict_factors(self):
        self.mask = None
        self.predict_factor = None

    def calculate_predict_factors(self, x0_start, x0_end, x1_start, x1_end, method="least_square", n=None):
        if self.ROI == (x0_start, x0_end, x1_start, x1_end) and self.predict_method == method and n == n:
            if self.predict_factor is not None:
                return

        self.ROI = (x0_start, x0_end, x1_start, x1_end)
        self.n = n
        self.predict_factor = None
        self.predict_method = method
        mask = np.zeros(shape=self.image_shape, dtype=np.bool)
        mask[x0_start:x0_end, x1_start:x1_end] = True
        self.mask = mask.flatten()

        if method == "scale_area":
            # self.scale_factor = 1.0/(1-np.array([np.sum(vec[mask]**2) for vec in self.V]))
            # self.scale_factor = 1.0/(1-np.array([linalg.norm(vec[mask]) for vec in self.V]))
            A = self.image_shape[0] * self.image_shape[1]
            self.predict_factor = A / (A - float((x0_end - x0_start) * (x1_end - x1_start)))
        elif method == "least_square":
            V_cp = np.copy(self.V[:n])
            V_cp[:, self.mask] = 0
            self.predict_factor = linalg.pinv(V_cp)
        else:
            raise ValueError("Method not found!! Avalialbe method:" + str(self.available_predict_method))

    def fit(self, data):
        self.image_shape = data[0].shape
        self.N = len(data)
        batch = np.zeros((self.N, self.image_shape[0] * self.image_shape[1]))
        for i in range(self.N):
            batch[i] = data[i].flatten()
        self.av = np.average(batch, axis=0)
        batch = batch - self.av
        U, self.S, self.V = linalg.svd(batch, full_matrices=0)
        self.predict_factor = None

    def predict_coeff(self, img, x0_start, x0_end, x1_start, x1_end, method="least_square", n=None):

        self.calculate_predict_factors(x0_start, x0_end, x1_start, x1_end, method=method, n=n)

        if self.image_shape != img.shape:
            raise ValueError("Image size is not correct!!")

        img_cp = img.copy().flatten()
        img_cp -= self.av
        img_cp[self.mask] = 0

        if method == "scale_area":
            c = np.dot(self.V[:n], img_cp.T) * self.predict_factor
        elif method == "least_square":
            c = np.dot(img_cp, self.predict_factor)
        else:
            assert False
        return c

    def predict(self, img, x0_start, x0_end, x1_start, x1_end, method="least_square", n=None):
        c = self.predict_coeff(img, x0_start, x0_end, x1_start, x1_end, method=method, n=n)
        return (np.dot(c, self.V[:n]) + self.av).reshape(self.image_shape)

    def save(self, fname):
        with open(fname, 'wb') as f:
            cPickle.dump(self.__dict__, f, 2)

    def load(self, fname):
        with open(fname, 'rb') as f:
            tmp_dict = cPickle.load(f)
        self.__dict__.update(tmp_dict)

    def remember_only(self, N):
        self.N = N
        self.V = self.V[:N]
        self.S = self.S[:N]


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

    def predict(self, shots, roi, basis="basis", plot=False, verbose=True, predict_method="least_square", n=None):
        if verbose:
            print "Running backup"
        self.backup(shots)
        roi_converted = roi_convert(roi)

        if verbose:
            print "Loading basis"

        if self.EE.av is None:  # No basis are loaded.
            self.EE.load(basis)

        print "Prediction Method: {0}".format(predict_method)
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
                    bg_predict = self.EE.predict(img_atom, *roi_converted, n=n, method=predict_method)
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
                        od_original_cp = np.copy(od_original)
                        od_new_cp = np.copy(od_new)
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
                        plt.title("Optimized, Method: {0}".format(predict_method))
                        plt.pcolormesh(x, y, od_new)
                        plt.xlim(0, img_atom.shape[1])
                        plt.ylim(0, img_atom.shape[0])
                        plt.colorbar()

                        plt.subplot("223")
                        error = np.sum((od_original_cp-1) ** 2)
                        plt.title("Original BG, error = {0:.2f}".format(error))
                        plt.pcolormesh(od_original_cp)
                        plt.colorbar()
                        plt.xlim(0, img_atom.shape[1])
                        plt.ylim(0, img_atom.shape[0])

                        plt.subplot("224")
                        error = np.sum((od_new_cp-1)**2)
                        plt.title("Optimized BG, error = {0:.2f}".format(error))
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
    parser.add_argument('--predict_method', action="store", dest='predict_method', type=str, default="least_square",
                        help= "Avalialbe method:" + str(EigenEngine.available_predict_method))
    args = parser.parse_args()
    shots = qrange.parse_range(args.range)
    shots = [int(shot) for shot in shots]
    n = args.n
    predict_method = args.predict_method

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
        EC.predict(shots, roi, plot=plot, predict_method=predict_method ,n=n)
