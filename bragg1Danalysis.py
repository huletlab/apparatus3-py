#!/usr/bin/python
import argparse
import os

# speed up the matplotlib
# ref: http://stackoverflow.com/questions/4931376/generating-matplotlib-graphs-without-a-running-x-server
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import *
from scipy.optimize import curve_fit
from configobj import ConfigObj

import qrange

magnif = 1.497  # um per pixel
lattice_d = 0.532

magnif = 1.497 / 1.03228  # um per pixel
lattice_d = 0.532

try:
    plt.style.use("ggplot")
except:
    pass


def double_gaussian(x, a, mu1, mu2, sigma, c):
    return -a * np.exp(-((x - mu1) ** 2) / 2 / sigma ** 2) + a * np.exp(-((x - mu2) ** 2) / 2 / sigma ** 2) + c


def process(diff):
    y = np.sum(diff, axis=0)
    x = np.arange(len(y))

    a_guess = (y.max() - y.min()) / 2
    mu1_guess, mu2_guess = float(y.argmin()), float(y.argmax())
    sigma_guess = np.abs(mu2_guess - mu1_guess) / 8
    p0 = [a_guess, mu1_guess, mu2_guess, sigma_guess, 0]

    try:
        p, q = curve_fit(double_gaussian, x, y, p0=p0)
    except:
        p = np.array(p0)
        print "Failed to fit ... use initial value instead!!"
    p = np.array(p)
    return p


try:
    from iopro import loadtxt

    np.loadtxt = loadtxt
except:
    pass


def bragg_1D_anlysis(datadir, shot, shot_ref, report=None, verbose=True, save_fig=True, rotate=False):
    shot_num = int(float(report['SEQ']['shot']))

    if not report:
        inifile = datadir + 'report' + "%04d" % shot + '.INI'
        report = ConfigObj(inifile)

    if not '1DBRAGG_ANALYSIS' in report.keys():
        report['1DBRAGG_ANALYSIS'] = {}
    try:
        ao0_freq = float(report['1DBRAGG']['AO0Freq'])
        ao1_freq = float(report['1DBRAGG']['AO1Freq'])

        delta_freq = ao0_freq - ao1_freq
        report['1DBRAGG_ANALYSIS']['DeltaFreq'] = delta_freq

    except Exception as e:
        if isinstance(shot, np.ndarray):
            print "ERROR getting value from INI file.  shot# = ", report['SEQ']['shot']
        else:
            print "ERROR getting value from INI file.  shot# = ", shot
        raise e

    if isinstance(shot, np.ndarray):
        cddata = shot
    else:
        cddata = np.loadtxt(os.path.join(datadir, '%04d_column.ascii' % shot))

    if isinstance(shot_ref, np.ndarray):
        cddata_ref = shot_ref
    else:
        cddata_ref = np.loadtxt(os.path.join(datadir, '%04d_column.ascii' % shot_ref))

    if rotate == True:
        cddata = np.rot90(cddata)

    diff = cddata - cddata_ref

    parameters = process(diff)

    onedprofile = diff.sum(axis=0)
    contrast = np.max(onedprofile) - np.min(onedprofile)
    a, mu1, mu2, sigma, c = parameters

    # num_scattering = a * np.sqrt(2 * pi * sigma)

    # shift_distance = (mu2 - mu1) * magnif

    y = np.sum(diff, axis=0)
    x = np.arange(len(y))
    y_fit = double_gaussian(x, *parameters)
    y_fit_nobg = double_gaussian(x, a, mu1, mu2, sigma, 0)

    y_positive = np.copy(y_fit_nobg)
    try:
        y_positive[y_fit_nobg < 0] = 0
    except:
        pass
    y_negative = np.copy(y_fit_nobg)
    try:
        y_negative[y_fit_nobg > 0] = 0
    except:
        pass

    num_scattering = np.sum(y_positive)

    com_p = np.average(x, weights=y_positive)
    com_n = np.average(x, weights=y_negative)

    shift_distance = (com_p - com_n) * magnif

    report['1DBRAGG_ANALYSIS']['NumScattering'] = num_scattering
    report['1DBRAGG_ANALYSIS']['ShiftDistance'] = shift_distance  # store in um
    report['1DBRAGG_ANALYSIS']['Amplitude'] = abs(a)
    report['1DBRAGG_ANALYSIS']['contrast'] = contrast
    report['1DBRAGG_ANALYSIS']['MomentumTransfer'] = num_scattering * shift_distance

    if verbose:
        if isinstance(shot, np.ndarray):
            to_print = "#{2}, NumScattering = {0:.0f}, ShiftDistance = {1:.2f} um".format(num_scattering,
                                                                                          shift_distance,
                                                                                          shot_num)
        else:
            to_print = "#{2} - #{3}, NumScattering = {0:.0f}, ShiftDistance = {1:.2f} um".format(num_scattering,
                                                                                                 shift_distance, shot,
                                                                                                 shot_ref)
        print(to_print)
    report.write()

    if save_fig:
        plt.figure(figsize=(5, 10), dpi=80)
        plt.subplot('211')
        if isinstance(shot, np.ndarray):
            plt.title("#{0}".format(shot_num) + r"$\Delta$ = {0}kHz".format(delta_freq * 1000))
        else:
            plt.title("#{0} - #{1}".format(shot, shot_ref) + r"$\Delta$ = {0}kHz".format(delta_freq * 1000))
        plt.imshow(diff)
        plt.colorbar()
        plt.subplot('212')
        plt.plot(x, y)
        plt.plot(x, y_fit,
                 label="NumScattering = {0:.0f}\nShiftDistance = {1:.2f} um".format(num_scattering, shift_distance))
        plt.ylim(y.min() * 1.3, y.max() * 1.3)
        plt.legend(loc='best').get_frame().set_alpha(0.5)

        plt.savefig('%04d_bragg_1D_analysis.png' % shot_num)
        plt.close()  # release the memory


def bragg_multi(datadir, shots, **kwargs):
    reports = [ConfigObj(datadir + 'report' + "%04d" % shot + '.INI') for shot in shots]
    cddatas = [np.loadtxt(os.path.join(datadir, '%04d_column.ascii' % shot)) for shot in shots]
    cddata_refs = []
    print "Searching for ref shots:"
    for i, report in enumerate(reports):
        if float(report['1DBRAGG']['AO0Freq']) == float(report['1DBRAGG']['AO1Freq']):
            print "#{0}".format(shots[i])
            cddata_refs.append(cddatas[i])
    cddata_ref = np.mean(cddata_refs, axis=0)

    print "Analysis begin:"
    for i, report in enumerate(reports):
        try:
            bragg_1D_anlysis(datadir, cddatas[i], cddata_ref, report=report, **kwargs)
        except Exception as err:
            print err


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--range', action="store", dest='range', help="analyze files in the specified range.")
    parser.add_argument('--shot', type=int)
    parser.add_argument('--ref', type=int)
    parser.add_argument('--no_save_fig', action='store_false')
    args = parser.parse_args()
    shot = args.shot
    shot_ref = args.ref
    save_fig = args.no_save_fig
    datadir = "./"

    if args.range:
        shots = qrange.parse_range(args.range)
        shots = [int(shot) for shot in shots]
        bragg_multi(datadir, shots, save_fig=save_fig, verbose=True)

    else:
        bragg_1D_anlysis(datadir, shot, shot_ref, save_fig=save_fig, verbose=True)
