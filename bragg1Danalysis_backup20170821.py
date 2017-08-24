#!/usr/bin/python
import argparse
import os

# speed up the matplotlib
# ref: http://stackoverflow.com/questions/4931376/generating-matplotlib-graphs-without-a-running-x-server
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np
from scipy.constants import *
from scipy.optimize import curve_fit
from configobj import ConfigObj
from scipy import ndimage
import qrange
import matplotlib.gridspec as gridspec
import os
import math
magnif = 1.497  # um per pixel
lattice_d = 0.532

magnif = 1.497 / 1.03228  # um per pixel
lattice_d = 0.532

try:
    plt.style.use("ggplot")
except:
    pass


def double_gaussian(x, a, mu1, mu2, sigma, c):
    return -a * np.exp(-((x - mu1) ** 2) / 2. / sigma ** 2) + a * np.exp(-((x - mu2) ** 2) / 2. / sigma ** 2) + c


def process(diff):
    y = np.sum(diff, axis=0)
    ## Symetrize the 1D profile before fitting
    #y = (y - y[::-1])
    #y = y[:len(y)/2]
    x = np.arange(len(y))

    a_guess = (y.max() - y.min())*0.5
    mu1_guess, mu2_guess = float(y.argmin()), float(y.argmax())
    sigma_guess = np.abs(mu2_guess - mu1_guess)/10 
    p0 = [a_guess, mu1_guess, mu2_guess, sigma_guess, 0]
    #param_bounds=([0,mu1_guess-20,mu2_guess-20,0,-5],[a_guess*2,mu1_guess+20,mu2_guess+20,len(y)/20.0,5]	)
    param_bounds=([0,mu1_guess-10,mu2_guess-10,0,-10],[a_guess*3,mu1_guess+10,mu2_guess+10,3,10]	)
    #param_bounds=([0,mu1_guess-10,mu2_guess-10,0,-5],[a_guess*5,mu1_guess+10,mu2_guess+10,100,10]	)
    try:
        p, q = curve_fit(double_gaussian, x, y, p0=p0,bounds=param_bounds)
        #p, q = curve_fit(double_gaussian, x, y, p0=p0)
    except:
    	p = [0, mu1_guess, mu2_guess, sigma_guess, 0]
        print "Failed to fit double gaussain... use default value instead!!"
    p = np.array(p)
    return p

def gaussian(x, a,c, sigma):
    return a * np.exp(-((x -c) ** 2) / 2 / sigma ** 2)

def process_ref(ref):
    y1 = np.sum(ref, axis=0)
    y2 = np.sum(ref, axis=1)
    ps=[]
    for y in [y1,y2]:	
	x = np.arange(len(y))
	a_guess = (y.max() - y.min())
	c_guess =  float(y.argmax())
	s_guess = len(x) / 2.0
	p0 = [a_guess,c_guess,s_guess]
	
	try:
		p, q = curve_fit(gaussian, x, y, p0=p0)
	except:
		p = np.array(p0)
		print "Failed to fit gaussian ... use initial value instead!!"

	p = np.array(p)
	ps.append(p)
		

    return ps

try:
    from iopro import loadtxt

    np.loadtxt = loadtxt
except:
    pass


def bragg_1D_anlysis(datadir, shot, shot_ref, report=None, verbose=False, save_fig=True, rotate=False, roi=None,smartroi=False,
                     section='1DBRAGG_ANALYSIS',fig_suffix="",key=None,rows = None,sumpos=[0,0,-1,-1],**kwargs):
    shot_num = int(float(report['SEQ']['shot']))

    if not report:
        inifile = datadir + 'report' + "%04d" % shot + '.INI'
        report = ConfigObj(inifile)

    if section not in report.keys():
        report[section] = {}
    try:
        ao0_freq = float(report['1DBRAGG']['AO0Freq'])
        ao1_freq = float(report['1DBRAGG']['AO1Freq'])
	tof = float(report['DIMPLELATTICE']['tof'])

        delta_freq = ao0_freq - ao1_freq
        report[section]['DeltaFreq'] = delta_freq

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

    if shot_ref is not None:
        if isinstance(shot_ref, np.ndarray):
            cddata_ref = shot_ref
        else:
            cddata_ref = np.loadtxt(os.path.join(datadir, '%04d_column.ascii' % shot_ref))
    else:
        cddata_ref = None

    if rotate:
        cddata = np.rot90(cddata)

    if cddata_ref is None:
        diff_original = cddata
    else:
        diff_original = cddata - cddata_ref

    if rows:
	diff = diff_original[rows[0]:rows[1]]
    else: 
	diff = diff_original
    parameters = process(diff)
    parameters_ref = process_ref(cddata_ref)
    a_ref_x, c_ref_x,sigma_ref_x = parameters_ref[0]
    a_ref_y, c_ref_y,sigma_ref_y = parameters_ref[1]

    onedprofile = diff.sum(axis=0)
    contrast = np.max(onedprofile) - np.min(onedprofile)
    a, mu1, mu2, sigma, c = parameters

    # num_scattering = a * np.sqrt(2 * pi * sigma)


    y = np.sum(diff, axis=0)
    x = np.arange(len(y))
    # y_original = np.sum(cddata, axis=0)
    # y_original = 1e4*np.exp(-(x-70.)**2/2.)
    y_original = np.sum(cddata, axis=0)
    x_original = np.arange(len(y_original))
    y_original_ref = np.sum(cddata_ref, axis=0)
    x_original_ref = np.arange(len(y_original_ref))
    y_fit_scale = 10
    x_fit = [ i*1./y_fit_scale for i in range(len(y)*y_fit_scale)]
    y_fit = double_gaussian(x_fit, *parameters)
    y_fit_nobg = double_gaussian(x_fit, a, mu1, mu2, sigma, 0)
    if not os.path.exists("./1D_analysis/{0}_{1}_{2}".format(key[0], key[1], report[key[0]][key[1]])):
		print "Creating 1D_analysis folder ./1D_analysis/{0}_{1}_{2}".format(key[0], key[1], report[key[0]][key[1]])
		os.makedirs("./1D_analysis/{0}_{1}_{2}".format(key[0], key[1], report[key[0]][key[1]]))

    np.savetxt('./1D_analysis/{0}_{1}_{2}/{3}kHz_{4}_{5}.dat'.format(key[0], key[1], report[key[0]][key[1]],"{0:04.1f}".format((delta_freq*1000)).replace(".","_"),shot_num,fig_suffix),y)
    # Try to get rid of symmetric part
    #y_fit_nobg = (y_fit_nobg[::-1] - y_fit_nobg)/2
    y_positive = np.copy(y_fit_nobg)
    y_positive -= sum(y_positive)*1./len(y_positive) 
    threshold = 0
    try:
        y_positive[y_positive < threshold] = 0
    except:
        pass
    y_negative = np.copy(y_fit_nobg)
    try:
        y_negative[y_fit_nobg > 0] = 0
    except:
        pass

	
#    num_scattering = (np.sum(y_positive)*np.sum(y_negative)*-1)**0.5/y_fit_scale 
    num_scattering = (np.sum(y_positive)*np.sum(y_negative)*-1)**0.5/y_fit_scale 
    #num_scattering = (np.sum(y_positive))/y_fit_scale 
    num_scattering_fit = a*sigma*(2*pi)**0.5
    #y_filter = np.copy(y)-np.mean(y)
    y_filter = ndimage.filters.gaussian_filter(y,1.0)
    y_filter = y_filter-np.mean(y_filter)
    #y_filter[abs(y_filter)<100]=0		
    com_diff = np.sum(np.multiply(x,y_filter))
    com_p = np.average(x_fit, weights=y_positive)
    com_n = np.average(x_fit, weights=y_negative)
    com_original = ndimage.measurements.center_of_mass(cddata)
    com_original_ref = ndimage.measurements.center_of_mass(cddata_ref)
    shift_distance = (com_p - com_n) * magnif
    y_original_positive = np.copy(y_original)
    y_original_positive[y_original_positive < 500] = 0
    com_original_positive = np.average(x_original, weights=y_original_positive)

    peakSep = (np.argmax(y_fit) - np.argmin(y_fit))*1./y_fit_scale* magnif
    peakLocPos = np.argmax(y_fit)*1./y_fit_scale* magnif
    peakLocNeg = np.argmin(y_fit)*1./y_fit_scale* magnif
    peakSep_raw = (np.argmax(y) - np.argmin(y)) * magnif
    ## This is simply wrong since the two center of the gausian will not be the peak and valley
    peakSep_fit = (mu2 - mu1) * magnif
#    a, mu1, mu2, sigma, c = parameters
    print shot_num,num_scattering
    report[section]['NumScattering'] = num_scattering
    report[section]['NumScattering_fit'] = num_scattering_fit
    report[section]['ShiftDistance'] = shift_distance  # store in um
    report[section]['peakSep'] = peakSep  # store in um
    report[section]['peakLocPos'] = peakLocPos  # store in um
    report[section]['peakLocNeg'] = peakLocNeg  # store in um
    report[section]['peakSep_raw'] = peakSep_raw  # store in um
    report[section]['peakSep_fit'] = peakSep_fit  # store in um
    report[section]['Amplitude'] = abs(a)
    report[section]['contrast'] = contrast
    report[section]['MomentumTransfer'] = num_scattering * shift_distance
    report[section]['OriginalCOMx'] = com_original[1]
    report[section]['OriginalCOMx_ref'] = com_original_ref[1]
    report[section]['OriginalCOMx_fit'] = c_ref_x
    report[section]['OriginalCOMx_delta'] = com_original[1]-com_original_ref[1]
    report[section]['DiffCOMx'] = com_diff
    report[section]['OriginalCOMx_positive'] = com_original_positive
    report[section]['OriginalCOMy'] = com_original[0]
    # report[section]['1dmax'] = np.max(y_original)
    # report[section]['1dmin'] = np.min(y_original)
    if smartroi:
	xmax=np.size(cddata,1)
	ymax=np.size(cddata,0)
	roi=np.zeros(4)
	roi[0]=max(c_ref_x-abs(sigma_ref_x*3),0)
	roi[2]=min(c_ref_x+abs(sigma_ref_x*3),xmax)
	roi[1]=max(c_ref_y-abs(sigma_ref_y*3),0)
	roi[3]=min(c_ref_y+abs(sigma_ref_y*3),ymax)

    if roi is not None:
    	#a_ref, c_ref,sigma_ref = parameters_ref
	roi=[int(roi[0]+com_original[1]),int(roi[1]+com_original[0]),int(roi[2]+com_original[1]),int(roi[3]+com_original[0])]
	#print roi
    	cddata_cut =  cddata[roi[1]:roi[3],roi[0]:roi[2]]
    	cddata_ref_cut =  cddata_ref[roi[1]:roi[3],roi[0]:roi[2]]
        report[section]['partial_sum'] = np.sum(cddata_cut)
    	cddata_diff_cut =  diff[roi[1]:roi[3],roi[0]:roi[2]]
        report[section]['partial_sum_diff'] = np.sum(cddata_diff_cut)
    	com_original_cut = ndimage.measurements.center_of_mass(cddata_cut)
    	com_original_ref_cut = ndimage.measurements.center_of_mass(cddata_ref_cut)
    	report[section]['OriginalCOMx_cut'] = com_original_cut[1]
    	report[section]['OriginalCOMx_delta_cut'] = com_original_cut[1] - com_original_ref_cut[1]
    else:
	cddata_cut =cddata
	cddata_diff_cut =diff

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
        gs = gridspec.GridSpec(3,2)
	gs.update(wspace=0.55,hspace=0.4)
	fig = plt.figure(figsize=(8, 12))
	ax = fig.add_subplot(gs[0, 0])
#        if key is not None:
#            ax.set_title("{0}:{1} = {2}".format(key[0], key[1], report[key[0]][key[1]]))
        ax.pcolor(cddata)
#        ax.plot([com_original[1]], [com_original[0]], 'wo')
        ax.set_xlim(0, cddata.shape[1])
        ax.set_ylim(0, cddata.shape[0])
        if roi is not None:
 		ax.add_patch(patches.Rectangle((roi[0],roi[1]), roi[2] - roi[0], roi[3] - roi[1],
                                                    facecolor="grey", fill=False))
            # ref: http://stackoverflow.com/questions/13013781/how-to-draw-a-rectangle-over-a-specific-region-in-a-matplotlib-graph
	ax_ref = fig.add_subplot(gs[0, 1])
        ax_ref.pcolor(cddata_ref)
        ax_ref.set_xlim(0, cddata_ref.shape[1])
        ax_ref.set_ylim(0, cddata_ref.shape[0])
        ax_ref.set_title("Reference")
#	if rows:
# 		ax_cut.add_patch(patches.Rectangle((0,rows[0]),cddata_cut.shape[1], rows[1]-rows[0],
#                                                    facecolor="grey",lw=2,ec='k', fill=False))

	ax2 = fig.add_subplot(gs[1, 0])
        if isinstance(shot, np.ndarray):
            ax2.set_title("#{0}".format(shot_num) + r"$\Delta$ = {0}kHz".format(delta_freq * 1000))
        else:
            ax2.set_title("#{0} - #{1}".format(shot, shot_ref) + r"$\Delta$ = {0}kHz".format(delta_freq * 1000))
        #plt.imshow(diff)
        ax2.pcolor(diff_original)
        ax2.plot([com_original[1]], [com_original[0]], 'wo')
        ax2.set_xlim(0, diff_original.shape[1])
        ax2.set_ylim(0, diff_original.shape[0])
#        if roi is not None:
# 		ax2.add_patch(patches.Rectangle((roi[0],roi[1]), roi[2] - roi[0], roi[3] - roi[1],
#                                                    facecolor="grey", fill=False))
            # ref: http://stackoverflow.com/questions/13013781/how-to-draw-a-rectangle-over-a-specific-region-in-a-matplotlib-graph
#	if rows:
# 		ax2.add_patch(patches.Rectangle((0,rows[0]),diff_original.shape[1], rows[1]-rows[0],
#                                                    facecolor="black",lw=2,ec='k', fill=False))
	ax2_cut = fig.add_subplot(gs[1, 1])
        ax2_cut.pcolor(cddata_diff_cut)
        ax2_cut.set_xlim(0, cddata_diff_cut.shape[1])
        ax2_cut.set_ylim(0, cddata_diff_cut.shape[0])
        ax2_cut.set_title("Cutted Region")

        #ax2.colorbar()

	ax3 = fig.add_subplot(gs[2, 0])

        ax3.plot(x, y, label="Difference")
	ax3.plot(x_fit, y_fit,label ='FitY\n'+ str(["%.2f"%i for i in [a, mu1, mu2, sigma, c]]))#+str(parameters))
        ax3.plot(x_original, y_original, label="Data")
        ax3.plot(x_original_ref, y_original_ref, label="Reference")
        ax3.axvline(c_ref_x, label="Reference Fitted Center",color="k")
	lengthOfX=len(x)
	nt=2
	dx=lengthOfX*1.0/(nt*2+1)
	xt=[c_ref_x+dx*(i-nt) for i in range(2*nt+1)] 
	lt = ["%.1f"%(abs(magnif*(x-c_ref_x))/tof*0.1) for x in xt]
	for a in [ax3]:
		a.set_xticks(xt)
    		a.set_xticklabels(lt)
#       ax3.plot(x_original, y_filter, label="Filtered")
#	ax3.plot(x, y_fit,
#                label="NumScattering = {0:.0f}\nShiftDistance = {1:.2f} um\npeakSep={2:.2f}um".format(num_scattering,
#                                                                                                       shift_distance,
#                                                                                                       peakSep))
#        ax3.axvline(com_original[1], color='deeppink', label="COM_2D")
#        ax3.axvline(c_ref_x, color='blue', label="COM_X_Fit")
        #ax3.axvline(sumXpos, color='k', label="sumXpos")
        y_original_positive = np.copy(y_original)
        y_original_positive[y_original_positive < 0] = 0
#        ax3.axvline(np.average(x_original, weights=y_original_positive), color='g', label="COM_1D_pos")
	ax3.set_xlabel("Velocity(cm/s)")
	ax3.set_ylabel("Atom Number")
        ax3.set_ylim(min(y_fit.min() * 1.3,-1000.0), y_original.max() * 1.3)
	ax3.legend(loc=2,bbox_to_anchor=(1.1, 1.05))
	ax4 = ax3.twiny()
	ax4.set_xlabel("Pixel")
	ax4.set_xlim(ax3.get_xlim())
	if not os.path.exists("./1D_analysis/{0}_{1}_{2}".format(key[0], key[1], report[key[0]][key[1]])):
		print "Creating 1D_analysis folder ./1D_analysis/{0}_{1}_{2}".format(key[0], key[1], report[key[0]][key[1]])
		os.makedirs("./1D_analysis/{0}_{1}_{2}".format(key[0], key[1], report[key[0]][key[1]]))
        fig.savefig('./1D_analysis/{0}_{1}_{2}/{3}kHz_{4}_{5}.png'.format(key[0], key[1], report[key[0]][key[1]],"{0:04.1f}".format((delta_freq*1000)).replace(".","_"),shot_num,fig_suffix))
#        print './1D_analysis/{0}_{1}_{2}/{3}kHz_{4}_{5}.png'.format(key[0], key[1], report[key[0]][key[1]],str(delta_freq*1000).replace(".","_"),shot_num,fig_suffix)
#	print delta_freq,delta_freq*1000,int(delta_freq*1000),str(delta_freq*1000)
        plt.close()  # release the memory
#    return contrast
    return num_scattering, contrast 

def bragg_multi_inner(datadir, reports, cddatas,nbgs=4,no_bragg_ref=0,number_filter=1.0,**kwargs):
    # reports = [ConfigObj(datadir + 'report' + "%04d" % shot + '.INI') for shot in shots]
    # cddatas = [np.loadtxt(os.path.join(datadir, '%04d_column.ascii' % shot)) for shot in shots]
    cddata_refs = {}
    #print "Searching for ref shots:"
    cddata_dict = {}
    cddata_dict_close_ref = {}
    shots  = [int(float(r["SEQ"]["shot"])) for r in reports]
    deltas  = [float(r['1DBRAGG']['AO0Freq']) - float(r['1DBRAGG']['AO1Freq'])  for r in reports]
    ref_number = []
    ref_shots = []
    averageN = np.average([ float(report['CPP']['nfit'])for i, report in enumerate(reports) if 'nfit' in report['CPP'] and 0<float(report['CPP']['nfit'])<1e9 ])
    print "Average number for all the shots:%.2e"%averageN
    upperNBound = averageN*number_filter+averageN
    lowerNBound = max(-averageN*number_filter+averageN,0)
    print "Filtering out number not in range %.3e<n<%.3e"%(lowerNBound,upperNBound)
    for i, report in enumerate(reports):
	if 'nfit' not in report['CPP']:
		print "Skipping %s shot with no nfit in CPP"%(int(float(report['SEQ']['shot'])))
		continue
	atomnumber = float(report['CPP']['nfit'])
	if not lowerNBound<atomnumber<upperNBound:
		print "Skipping %s shot with N=%.3e"%(int(float(report['SEQ']['shot'])),atomnumber)
		continue
        delta = float(report['1DBRAGG']['AO0Freq']) - float(report['1DBRAGG']['AO1Freq'])
	if (not no_bragg_ref and float(report['DIMPLELATTICE']['braggkill'])==1.0) or float(report['DIMPLELATTICE']['braggkill'])==1.0:
        	if not delta in cddata_dict:
			cddata_dict[delta] = []
        	cddata_dict[delta].append(cddatas[i])
        if float(report['1DBRAGG']['AO0Freq']) == float(report['1DBRAGG']['AO1Freq']):
		if  no_bragg_ref and float(report['DIMPLELATTICE']['braggkill'])!=0:
			continue
		if  not no_bragg_ref and float(report['DIMPLELATTICE']['braggkill'])==0:
			continue
		#print "#{0}".format(shots[i])
		cddata_refs[shots[i]]=cddatas[i]
		ref_number += float(report['CPP']['nfit']),
		ref_shots += float(report['SEQ']['shot']),
    cddata_ref_dict = {}
    cddata_ref_dic_delta = {}
    for i,s in enumerate(shots):
	refsN=min(nbgs,len(cddata_refs.keys()))
	refshots=[]
	refs= cddata_refs.keys()
	for n in range(refsN):
		ref_shot = min(refs,key=lambda x:abs(x-s))
		refs.remove(ref_shot)
		refshots.append(ref_shot)
	#print "Using ",refshots, " as ref for #%04d"%(s)
	
        cddata_ref_dict[int(s)] = cddata_refs[refshots[0]]/refsN
	for ref in refshots[1:]:
        	cddata_ref_dict[int(s)] = cddata_ref_dict[int(s)]+cddata_refs[ref]/refsN
	if deltas[i] in cddata_ref_dic_delta:
		cddata_ref_dic_delta[deltas[i]] += [cddata_ref_dict[int(s)]]
	else:
		cddata_ref_dic_delta[deltas[i]] = [cddata_ref_dict[int(s)]]

    # Calucutate average background for each Frequency
    for i in cddata_ref_dic_delta:
	refsN = len(cddata_ref_dic_delta[i])
	temp = cddata_ref_dic_delta[i][0]/refsN
	for ref in cddata_ref_dic_delta[i][1:]:
        	temp = temp +ref/refsN
	cddata_ref_dic_delta[i] = temp
    ref_ave_num = sum(ref_number)/len(ref_number)
    ref_std_num = np.std(ref_number)
    ref_fluctuation = ref_std_num/ref_ave_num*100.0
    print "Ref Atom number ave: %f std:%f fluctuation:%f percent"%(ref_ave_num,ref_std_num,ref_fluctuation)
    print "Analysis begin:"

 #   for i, report in enumerate(reports):
 #       try:
 #           bragg_1D_anlysis(datadir, cddatas[i], cddata_ref_dict[shots[i]], report=report, **kwargs)
 #       except Exception as err:
 #           print "Shot",shots[i]," fail the analysis, probably need to remove this shot."
 #           print err

#### Start porcessing the averaging part

    kwargs['save_fig'] = kwargs['save_ave'] or kwargs['save_fig']
    cddata_mean_dict = {}
    cddata_mean_1D_dict = {}
    cddata_mean_1D_max = []
    cddata_mean_1D_min = []
    cddata_mean_1D_max_value = []
    cddata_mean_1D_min_value = []
    cddata_mean_ref = np.mean(cddata_refs.values(), axis=0)	
    parameters_ref = process_ref(cddata_mean_ref)
    a_ref_x, c_ref_x,sigma_ref_x = parameters_ref[0]
    a_ref_y, c_ref_y,sigma_ref_y = parameters_ref[1]
    com_mean_ref = ndimage.measurements.center_of_mass(cddata_mean_ref)
    cddata_mean_ref_1D = np.sum(cddata_mean_ref, axis=0)	
    for key, cddata_list in cddata_dict.iteritems():
#	if key ==0:
#		lenOfRefs  = len(cddata_list)
#		poolOfRefs = max(1,lenOfRefs/2)
#		print "Processing Ref. Threr are total %d refs. Use %d shots for 0Hz Data"%(lenOfRefs,poolOfRefs)
#        	cddata_mean_dict[key] = np.mean(cddata_list[-poolOfRefs:], axis=0)
#	else:
#        	cddata_mean_dict[key] = np.mean(cddata_list, axis=0)
        cddata_mean_dict[key] = np.mean(cddata_list, axis=0)
        cddata_mean_1D_dict[key] =  np.sum(cddata_mean_dict[key], axis=0)-cddata_mean_ref_1D
        cddata_mean_1D_max.append([key,np.argmax(cddata_mean_1D_dict[key])])
        cddata_mean_1D_max_value.append([key,np.max(cddata_mean_1D_dict[key])])
        cddata_mean_1D_min.append([key,np.argmin(cddata_mean_1D_dict[key])])
        cddata_mean_1D_min_value.append([key,np.min(cddata_mean_1D_dict[key])])
    cddata_mean_1D_max=np.array(cddata_mean_1D_max)
    cddata_mean_1D_min=np.array(cddata_mean_1D_min)
    cddata_mean_1D_max_value=np.array(cddata_mean_1D_max_value)
    cddata_mean_1D_min_value=np.array(cddata_mean_1D_min_value)
    
    for n,freq in enumerate(sorted(cddata_mean_1D_dict)):
	if n ==0 :
		cddata_mean_1D = [cddata_mean_1D_dict[freq]]
	else:
		cddata_mean_1D = np.append(cddata_mean_1D, [cddata_mean_1D_dict[freq]],axis=0)

    deltas = []
    excitation=[]
    contrast=[]
    tof = 0.0 
    for i, report in enumerate(reports):
	delta = float(report['1DBRAGG']['AO0Freq']) - float(report['1DBRAGG']['AO1Freq'])
	if i ==0:
		tof = float(report['DIMPLELATTICE']['tof'])
	if delta not in deltas and (not no_bragg_ref or float(report['DIMPLELATTICE']['braggkill'])==1.0):
    		deltas.append(delta)	
		try:
            		ex=bragg_1D_anlysis(datadir, cddata_mean_dict[delta], cddata_ref_dic_delta[delta] , report=report,
                             section='1DBRAGG_ANALYSIS_AVERAGED',fig_suffix="_average",
                             **kwargs)
            		#ex=bragg_1D_anlysis(datadir, cddata_mean_dict[delta], cddata_mean_ref, report=report,
                        #     section='1DBRAGG_ANALYSIS_AVERAGED',fig_suffix="_average",
                        #     **kwargs)
			excitation.append(ex[0])
			contrast.append(ex[1])
		except Exception as err:
			print err,"!!!!!!!!!!!"
			excitation.append(0)
			contrast.append(0)
    gs = gridspec.GridSpec(3,3)
    gs.update(wspace=0.55,hspace=0.4)
    fig = plt.figure(figsize=(12, 12))
    fig.suptitle("Ref Atom number ave: %.2e std:%.2e fluctuation:%.1f percent"%(ref_ave_num,ref_std_num,ref_fluctuation),fontsize=20)
    axNumber = fig.add_subplot(gs[0, 2])
    axNumber.plot(ref_shots,ref_number)
    axNumber.set_xlabel("Ref Shot Number")
    axNumber.set_ylabel("Atom Number")


    ax1 = fig.add_subplot(gs[0, 0])
    ax1.pcolor(cddata_mean_1D)
    ax1.set_xlim(0, cddata_mean_1D.shape[1])
    ax1.set_ylim(0, cddata_mean_1D.shape[0])
    ax1.set_ylabel("Bragg Frequency (kHz)")
    ax1.set_xlabel("Velocity(cm/s)")
    ax1.set_title("Excitation Spectrum")
    #ax.axvline(com_mean_ref[1],color="k")
    ax1.axvline(c_ref_x,color="k",label="Fitted Gaussian Center")
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.pcolor(cddata_mean_1D.clip(max=0))
    #ax2.axvline(com_mean_ref[1],color="k")
    ax2.axvline(c_ref_x,color="k",label="Fitted Gaussian Center")
    ax2.set_xlim(0, cddata_mean_1D.shape[1])
    ax2.set_ylim(0, cddata_mean_1D.shape[0])
    ax2.set_ylabel("Bragg Frequency (kHz)")
    ax2.set_xlabel("Velocity(cm/s)")
    ax2.set_title("Excitation Spectrum Absorption")
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.pcolor(cddata_mean_1D.clip(min=0))
    #ax3.axvline(com_mean_ref[1],color="k")
    ax3.axvline(c_ref_x,color="k",label="Fitted Gaussian Center")
    ax3.set_xlim(0, cddata_mean_1D.shape[1])
    ax3.set_ylim(0, cddata_mean_1D.shape[0])
    ax3.set_ylabel("Bragg Frequency (kHz)")
    ax3.set_xlabel("Velocity(cm/s)")
    ax3.set_title("Excitation Spectrum Emission")
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.plot(cddata_mean_1D_max[:,1],cddata_mean_1D_max[:,0]*1e3,".r",label="Emission")
    ax4.plot(cddata_mean_1D_min[:,1],cddata_mean_1D_min[:,0]*1e3,".b",label="Absorption")
    ax4.axvline(c_ref_x,color="k",label="Fitted Gaussian Center")
    #print "COM of reference ",(com_mean_ref[1],com_mean_ref[0])
    # 1.47um-1 is the default delta K	
    x =  np.linspace(0,c_ref_x,100)
    #print x
    magnif = 1.497
    ax4.plot(x,magnif*abs(x-c_ref_x)*1.47/2./3.14/tof,"-g")
    ax4.set_xlim(0, cddata_mean_1D.shape[1])
    ax4.set_ylim(0, cddata_mean_1D.shape[0])
    ax4.set_ylabel("Bragg Frequency (kHz)")
    ax4.set_xlabel("Velocity(cm/s)")
    ax4.set_title("Excitation Peak Location")
    ax4.legend(bbox_to_anchor=[1.2,1.0], loc='upper left')
    
    ax5 = fig.add_subplot(gs[0,1])
#    ax5.axvline(c_ref_x,color="k",label="Fitted Gaussian Center")
#    print len(deltas)
#    print len(excitation)
    ax5.plot(np.array(deltas)*1e3,excitation,".r",label="Excitation")
    #ax5.plot(cddata_mean_1D_max_value[:,0]*1e3,(cddata_mean_1D_max_value[:,1]-cddata_mean_1D_min_value[:,1])*0.5,".r",label="Excitation")
#    ax5.plot(,".r",label="Excitation")
    #ax5.plot(cddata_mean_1D_min_value[:,0]*1e3,abs(cddata_mean_1D_min_value[:,1]),".b",label="Absorption")
    #ax5.plot(np.sum(cddata_mean_1D.clip(min=0),axis=1))
#    ax5.set_xlim(0, cddata_mean_1D.shape[0])
    #ax5.set_ylim(0, cddata_mean_1D.shape[0])
    ax5.set_xlabel("Bragg Frequency (kHz)")
    ax5.set_ylabel("Excitation")
    ax5.set_title("Excitation")
#    ax5.set_title("Emission  Summed")
    ax6 = fig.add_subplot(gs[1,1])
    ax6.plot(np.array(deltas)*1e3,contrast,".r",label="Contrast")
    ax6.set_xlabel("Bragg Frequency (kHz)")
    ax6.set_ylabel("Contrast")
    ax6.set_title("Contrast")
#    ax6.axvline(c_ref_x,color="k",label="Fitted Gaussian Center")
#    ax6.plot(cddata_mean_ref_1D,".k",label="Reference Shot")
#    ax6.set_xlabel("Piexel")
#    ax6.set_xlabel("Linear Density")
#    ax6.set_title("Reference Shot")
#    fig.canvas.draw()
#    magnif = 1.497
#    tof
    lengthOfX=cddata_mean_1D.shape[1]
    nt=2
    dx=lengthOfX*1.0/(nt*2+1)
    xt=[c_ref_x+dx*(i-nt) for i in range(2*nt+1)] 
    lt = ["%.1f"%(abs(magnif*(x-c_ref_x))/tof*0.1) for x in xt]
    for ax in [ax1,ax2,ax3,ax4]:
	ax.set_xticks(xt)
    	ax.set_xticklabels(lt)
    key = kwargs["key"]
    if not os.path.exists("./1D_analysis/{0}_{1}_{2}".format(key[0], key[1], reports[0][key[0]][key[1]])):
		print "Creating 1D_analysis folder ./1D_analysis/{0}_{1}_{2}".format(key[0], key[1], reports[0][key[0]][key[1]])
		os.makedirs("./1D_analysis/{0}_{1}_{2}".format(key[0], key[1], reports[0][key[0]][key[1]]))
    
    fig.savefig('./1D_analysis/{0}_{1}_{2}_summary.png'.format(key[0], key[1], reports[0][key[0]][key[1]]))
    np.savetxt('./1D_analysis/{0}_{1}_{2}_excitation.txt'.format(key[0], key[1], reports[0][key[0]][key[1]]),(np.array(deltas)*1e3,np.array(excitation)))
    np.savetxt('./1D_analysis/{0}_{1}_{2}_contrast.txt'.format(key[0], key[1], reports[0][key[0]][key[1]]),(np.array(deltas)*1e3,np.array(contrast)))

def exist_shot(datadir, shots):
    for shot in shots:
        if os.path.isfile(os.path.join(datadir, '%04d_column.ascii' % shot)):
            yield shot


def bragg_multi(datadir, shots, key=None,no_bragg_ref=0, **kwargs):
    shots = [shot for shot in exist_shot(datadir, shots)]
    reports = [ConfigObj(datadir + 'report' + "%04d" % shot + '.INI') for shot in shots]
    cddatas = [np.loadtxt(os.path.join(datadir, '%04d_column.ascii' % shot)) for shot in shots]
    if key is None:
	print "Please Specify a key"
        #bragg_multi_inner(datadir, reports, cddatas, **kwargs)
    else:
        report_group_dict = dict()
        cddata_group_dict = dict()
        density_group_dict = dict()
        for i, report in enumerate(reports):
            group_key = report[key[0]][key[1]]
            if group_key not in report_group_dict:
                report_group_dict[group_key] = []
                cddata_group_dict[group_key] = []
                density_group_dict[group_key] = []
            
            if float(report["DIMPLELATTICE"]["tof"]) == 0:
            	density_group_dict[group_key].append(int(float(report["SEQ"]["shot"])))
            else:
            	report_group_dict[group_key].append(report)
            	cddata_group_dict[group_key].append(cddatas[i])

        for group_key in report_group_dict.keys():
            print '==============================================='
            print "Group: {0}:{1} = {2}".format(key[0], key[1], group_key)
            bragg_multi_inner(datadir, report_group_dict[group_key], cddata_group_dict[group_key], key=key, no_bragg_ref=no_bragg_ref,**kwargs)
        
	for group_key in report_group_dict.keys():
            print "Group: {0}:{1} = {2}".format(key[0], key[1], group_key)
            print "Density shots are:", ",".join(["%04d"%i for i in density_group_dict[group_key]])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--range', action="store", dest='range', help="analyze files in the specified range.")
    parser.add_argument('--shot', type=int)
    parser.add_argument('--ref', type=int)
    parser.add_argument('--nbgs', type=int,help="number of backgrounds to use when calculating diff column density",default=4)
    parser.add_argument('--no_save_fig', action='store_false')
    parser.add_argument('--no_bragg_ref', action='store_true',help="Whether if the refs having braggkill on")
    parser.add_argument('--save_ave', action='store_true',help="save fig for average shots")
    parser.add_argument('--smartROI', action='store_true')
    parser.add_argument('--ROI', action="store", dest='roi', help="")
    parser.add_argument('--rows', action="store", dest='rows', help="")
    parser.add_argument('--number_filter', action="store", dest='nfilter', help="Ignore shots with number more than x%",default='30.0')
    parser.add_argument('--key', action='store', dest='key', help="section:key", default=None)

    args = parser.parse_args()
    shot = args.shot
    shot_ref = args.ref
    save_fig = args.no_save_fig
    save_ave = args.save_ave
    key = args.key
    
    if key is not None:
        key = key.split(':')

    datadir = "./"
    roi = args.roi
    smartroi = args.smartROI
    if roi is not None:
        roi = roi.split(",")
		
        roi = [int(i.replace("m","-")) for i in roi]
    
    rows = args.rows
    if rows is not None:
        rows = map(int,rows.split(","))
		

    if args.range:
        shots = qrange.parse_range(args.range)
        shots = [int(shot) for shot in shots]
        bragg_multi(datadir, shots, save_fig=save_fig, save_ave=save_ave,verbose=False, roi=roi,smartroi=smartroi, key=key,rows=rows,nbgs=args.nbgs,no_bragg_ref=args.no_bragg_ref,number_filter=float(args.nfilter)*0.01)

    else:
        bragg_1D_anlysis(datadir, shot, shot_ref, save_fig=save_fig, verbose=False, roi=roi,smartroi=smartroi)
