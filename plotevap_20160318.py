#!/usr/bin/python

import argparse
import fitlibrary
import datetime
import pprint
import configobj
import uncertainties as unc
import re
import copy

import numpy
import math
import scipy

import matplotlib.pyplot as plt
import matplotlib

import subprocess
import os
import sys
sys.path.append('/lab/software/apparatus3/bin/py')

import statdat
import qrange

keys = ['SEQ:shot', 'ODT:odttof', 'EVAP:image', 'EVAP:finalcpow', \
        'ODTCALIB:maxdepth', 'ODTCALIB:v0radial', 'ODTCALIB:v0axial', \
        'CPP:nfit', 'CPP:peakd', 'CPP:ax0w', 'CPP:ax1w', \
        'CPP:TF', 'CPP:TF_az', 'CPP:T_az', 'CPP:T_2d_rd', \
        'CPP:T_2d_ax', 'CPP:TF_2d', 'FESHBACH:bias', \
        'EVAP:fieldrampfinal']

k={} # This dictionary provides numeric indices to all the keys
for i,key in enumerate(keys):
  k[key] = i

#colors = ['red','green', 'blue', 'black', 'magenta', 'cyan', 'yellow', 'orange', 'firebrick', 'steelblue']
colors = ['black', 'brown', 'red', 'orange', 'green', 'blue', 'magenta', 'gray', 'gold', \
          'black', 'brown', 'red', 'orange', 'green', 'blue', 'magenta', 'gray', 'gold', \
          'black', 'brown', 'red', 'orange', 'green', 'blue', 'magenta', 'gray', 'gold']

# Points that have error bars larger than this will not be shown
# value is as fraction of the data point
maxerror = .75

#--------------------------------------------------
#   EXTRACT SAME 
#
#   This looks at all the shots in the range and then
#   separates them into groups according to the given
#   parameter
#--------------------------------------------------
def extract_same( evapdat , parameter='EVAP:finalcpow'):
  print "Extracting Same, with respect to %s" % parameter
  evapdat = evapdat.tolist()
  pvalues = []
  extracted = {}
  for row in evapdat:
     value = row[k[parameter]]
     # These other values below can be used to 
     # differentiate shots that are different
     # but have the same parameter value 
     fcpow = row[k['EVAP:finalcpow']]
     U0    = row[k['ODTCALIB:maxdepth']]
     image = row[k['EVAP:image']]
     
     # If the parameter is recognized as a
     # known one, then a more physics friendly
     # value can be calculated and used instead
     if parameter == 'EVAP:finalcpow':
       U = value /10. * U0
       depth = U - 1e-7*image
       value = depth
     elif parameter == 'FESHBACH:bias':
       field = 6.8 * value
       value = field
     elif parameter == 'EVAP:fieldrampfinal':
       field = 6.8 * value
       value = field

     if value  in pvalues:
       l = extracted[value] 
       l.append ( list(row) )
       extracted[value] = l 
     else:
       pvalues.append( value )
       extracted[ value ] = [ list(row) ]
  return extracted
  
  #pprint.pprint( depths )
  #pprint.pprint( extracted )

#--------------------------------------------------
#   PARAMETER VALUE FOR PLOTTING
#
#   In several plots, the plot parameter value is 
#   required. This functions returns a numpy array
#   with the parameter value, which is ready to use
#   for plotting with matplotlib
def pvalue( points, parameter ):
  xArray = points[0,k[parameter]]
  if parameter == 'EVAP:image':
    xArray = xArray / 1000.
  if parameter == 'FESHBACH:bias':
    xArray = xArray *6.8
  if parameter == 'EVAP:fieldrampfinal':
    xArray = xArray *6.8
  xT = numpy.array( [ xArray ] )
  return xT
#--------------------------------------------------
#   FERMI TEMPERATURE
#
#   This gets the Fermi Temperature from the trap
#   depth and the full depth trap frequencies
#   Fermi Energy = h vbar (3N)^1/3
#  
#   where N is total atom number (i.e. counting
#   both spin states)
#
#   For the number, N, it uses the average of
#   the shots that are available, which are at
#   various time-of-flight's. 
#--------------------------------------------------
def t_fermi( points, ax_N, xT):
  h = 48. # This is h/kb in uK/MHz
  
  v = points[0,[k['EVAP:finalcpow'],k['ODTCALIB:v0radial'],k['ODTCALIB:v0axial'] ]].tolist()
  fraction = v[0]/10. 
  vradial  = v[1]*math.sqrt( fraction ) * 1e-6 # radial trap freq in MHz
  vaxial   = v[2]*math.sqrt( fraction ) * 1e-6 # axial trap freq in MHz 

  numbers = points[:,k['CPP:nfit']]
  numbers = numpy.array ( [ n for j,n in enumerate(numbers) if n>10000. and n<1e8  ] )
  num = unc.ufloat( (numpy.mean( numbers ) , scipy.stats.sem( numbers )) ) / 1e5
 
  densities = points[:,[k['ODT:odttof'],k['CPP:peakd']]]
  dens=[]
  for row in densities:
    if row[0] == 0.0:
      dens.append(numpy.asscalar(row[1]))
  if len(dens) == 0:
    den = unc.ufloat (( 0.0, 0.0 ) )
  elif len(dens) == 1:
    den = unc.ufloat( ( dens[0]  , 0.0 ) ) /1e12
  else:
    dens = numpy.array( dens )
    den = unc.ufloat( (numpy.mean( dens ) , scipy.stats.sem( dens )) ) /1e12

  try: 
    tf = h * ( vradial * vradial * vaxial * 3 * num * 1e5 )**(1./3.)
  except:
    tf = unc.ufloat( ( 0.0001, 0.0 ) ) ;

  #******* PLOT NUMBER AS A FUNCTION OF PARAMETER
  yT = numpy.array( [num.nominal_value] )
  yTerr = numpy.array( [num.std_dev()] )
  if yTerr[0] < maxerror* yT[0]:
      ax_N[0].errorbar(xT, yT, yerr=yTerr, fmt='s', ecolor=colors[i], markeredgecolor=colors[i], markerfacecolor="None", markeredgewidth=2., markersize=8, alpha=1.0)

  #******* PLOT DENSITY AS A FUNCTION OF PARAMETER
  yT = numpy.array( [den.nominal_value] )
  yTerr = numpy.array( [den.std_dev()] )
  if yTerr[0] < maxerror* yT[0]:
      ax_N[1].errorbar(xT, yT, yerr=yTerr, fmt='o', ecolor=colors[i], markeredgecolor=colors[i], markerfacecolor="None", markeredgewidth=2., markersize=8, alpha=1.0)

  return  tf, num, den
  #print points[:,[k['EVAP:finalcpow']  ]]
#  vradial = 

#--------------------------------------------------
#   TOF TEMPERATURE
#
#   This extracts the temprature from time-of-flight
#   data.
#--------------------------------------------------
def tof_temperature( points, tfermi, number, i, ax_tof, ax_T, ax_TF, axTFN, legend_dict, xT): 

  shots  = points[:,k['SEQ:shot']] 
  fcpow = points[0,k['EVAP:finalcpow']]
  U0    = points[0,k['ODTCALIB:maxdepth']]
 
  tofdat = points[:,k['ODT:odttof']]
  sizdat = points[:,k['CPP:ax0w']]
  
  mintofindex = tofdat.argmin()
  s_guess = sizdat[mintofindex]
  T_guess =fcpow/10. *U0 / 5.
  print "Fitting TOF Temperature, starting with guess r0 = %.3f, T = %.3f" % (s_guess, T_guess)
  p0 = [ s_guess, T_guess] 
  fitfun = fitlibrary.fitdict['Temperature'].function


  todelete=[]
  tofdat_clean =[]
  sizdat_clean =[]
  for j,tof in enumerate(tofdat):
    if sizdat[j] > 400. or sizdat[j] < 0.:
      todelete.append(j)
    else:
      tofdat_clean.append( tofdat[j] )
      sizdat_clean.append( sizdat[j] ) 

  tofdat = numpy.array(tofdat_clean)
  sizdat = numpy.array(sizdat_clean)

  fitdat= numpy.transpose(numpy.array( [tofdat.tolist(), sizdat.tolist()] ))
  
  try: 
    Tfit, Terror = fitlibrary.fit_function( p0, fitdat, fitfun)
    #getTcmd = 'getTrange -T %.2f %04d:%04d' % (depth/5., sorted(shots)[0], sorted(shots)[-1] )
    #getT = subprocess.Popen(getTcmd.split(), stdout=subprocess.PIPE).communicate()[0]
  except:
    Tfit =  [0., 0.]
    Terror = [0., 0.]
 
  #print points[:, [k['ODT:odttof'],k['CPP:ax0w']]]
  #print "Python Fit:  T = %.3f +/- %.3f uK " % (T, Terr)
  #print "Gnuplot Fit: T = %.3f +/- %.3f uK " % (float(getT.split()[3]), float(getT.split()[4]))
  
  T = unc.ufloat(( Tfit[1], Terror[1]))
 
  TfitX, TfitY = fitlibrary.plot_function( Tfit, tofdat, fitfun, xlim=(0.,6.0))
  ax_tof.plot( tofdat, sizdat, 'o', color=colors[i], markeredgewidth=0.8, markersize=4)
  ax_tof.plot( TfitX, TfitY, '-', color=colors[i], markeredgewidth=0.3, markersize=12, alpha=0.5)

  #******* T TOF
  #xT = numpy.array( [points[0,k['EVAP:image']]/1000. ] )
  yT = numpy.array( [T.nominal_value] )
  yTerr = numpy.array( [T.std_dev()] )
  if yTerr[0] < maxerror* yT[0]:
    for ax in ax_T:
      ax.errorbar(xT, yT, yerr=yTerr, fmt='o', ecolor=colors[i], markeredgecolor=colors[i], markerfacecolor="None", markeredgewidth=2., markersize=8, alpha=1.0)

  #******* T/TF TOF
  TF = T/tfermi
  yT = numpy.array( [TF.nominal_value] )
  yTerr = numpy.array( [TF.std_dev()] )
  if yTerr[0] < maxerror* yT[0]:
    for ax in ax_TF:
      plot1 = ax.errorbar(xT, yT, yerr=yTerr, fmt='o', ecolor=colors[i], markeredgecolor=colors[i], markerfacecolor="None", markeredgewidth=2., markersize=8, alpha=1.0)
      if 'Ballistic Expansion' not in legend_dict.keys():
        legend_dict['Ballistic Expansion'] = plot1

  #******* T/TF vs. N TOF
  xT = numpy.array( [number.nominal_value] )
  yT = numpy.array( [TF.nominal_value] )
  xTerr = numpy.array( [number.std_dev()] ) 
  yTerr = numpy.array( [TF.std_dev()] )
  if yTerr[0] < maxerror* yT[0]:
    axTFN.errorbar(xT,yT, xerr=xTerr, yerr=yTerr, fmt='o', ecolor=colors[i], markeredgecolor=colors[i], markerfacecolor="None", markeredgewidth=2., markersize=8, alpha=1.0)


  return T, TF, TfitX, TfitY


#--------------------------------------------------
#   TEMPERATURES FROM AZIMUTHAL FITS
#
#   From the polylog fit to azimuthally averaged 
#   data one can extract T/TF directly from the 
#   fugacity or indirectly through determination
#   of T from the size+trapfreqs and TF from the
#   number.
#
#   This function returns both the fugacity and
#   size estimates of T/TF obtained from the
#   azimuthally averaged data.
#--------------------------------------------------
def t_azimuthal( points , tfermi, number, i, ax_tof, ax_T, ax_TF, axTFN, legend_dict, xT):
  TF = numpy.mean(points[:,k['CPP:TF']])

  tof = points[:,k['ODT:odttof']]
  TF_az_fug = [] # T/TF from azimuthal fugacity
  T_az_size = [] # T from azimuthal sizes
  for j,t in enumerate(tof):
    if t > 0.0 and points[j,k['CPP:TF_az']] < 1.e5: 
      TF_az_fug.append(  points[j,k['CPP:TF_az']] )
      T_az_size.append( points[j,k['CPP:T_az']]  ) 
  
  print TF_az_fug 
 
  TF_az_f = unc.ufloat( (numpy.mean( TF_az_fug), numpy.std( TF_az_fug ) ) )
   
  # To obtain the indirect estimate, the Fermi temperature
  # obtained from the average of all shots at this trap depth
  # is used.  This value is passed into this function as tfermi
  TF_az_s = unc.ufloat( (numpy.mean( T_az_size), numpy.std( T_az_size ) ) ) / tfermi
  T_az_s = unc.ufloat( (numpy.mean( T_az_size), numpy.std( T_az_size ) ) ) 
 
  #print "Iteration = ", i
  #print "Color     = %s" % colors[i] 
  
  ax_tof.plot( tof, points[:,k['CPP:TF_az']], 'D', markeredgecolor=colors[i], markerfacecolor="None", markeredgewidth=1.0, markersize=3, alpha=1.0)
  ax_tof.plot( tof, points[:,k['CPP:T_az']]/tfermi.nominal_value, 'x', markeredgecolor=colors[i], markerfacecolor="None", markeredgewidth=1.0, markersize=3, alpha=1.0)
 
  evapimage = points[0,k['EVAP:image']]/1000.

  #******* T FUGACITY
  #xT = numpy.array( [ evapimage ] )
  T_az_f = TF_az_f * tfermi
  yT = numpy.array( [T_az_f.nominal_value ] )
  yTerr = numpy.array( [T_az_f.std_dev() ] )
  if yTerr[0] < maxerror* yT[0]:
    for ax in ax_T:
      ax.errorbar(xT, yT, yerr=yTerr, fmt='D', ecolor=colors[i], markeredgecolor=colors[i], markerfacecolor="None", markeredgewidth=2., markersize=8, alpha=1.0)

  #******* T/TF FUGACITY
  yT = numpy.array( [TF_az_f.nominal_value ] )
  yTerr = numpy.array( [TF_az_f.std_dev() ] )
  if yTerr[0] < maxerror* yT[0]:
    for ax in ax_TF:
      plot1 = ax.errorbar(xT, yT, yerr=yTerr, fmt='D', ecolor=colors[i], markeredgecolor=colors[i], markerfacecolor="None", markeredgewidth=2., markersize=8, alpha=1.0)
      if 'Azimuthal Fit Fugacity' not in legend_dict.keys():
        legend_dict['Azimuthal Fit Fugacity'] = plot1

  #******* T SIZE
  yT = numpy.array( [T_az_s.nominal_value ] )
  yTerr = numpy.array( [T_az_s.std_dev() ] )
  if yTerr[0] < maxerror* yT[0]:
    for ax in ax_T:
      ax.errorbar(xT, yT, yerr=yTerr, fmt='x', ecolor=colors[i], markeredgecolor=colors[i], markerfacecolor="None", markeredgewidth=2., markersize=8, alpha=1.0)

  #******* T/TF SIZE
  yT = numpy.array( [TF_az_s.nominal_value ] )
  yTerr = numpy.array( [TF_az_s.std_dev() ] )
  if yTerr[0] < maxerror* yT[0]:
    for ax in ax_TF:
      plot1 = ax.errorbar(xT, yT, yerr=yTerr, fmt='x', ecolor=colors[i], markeredgecolor=colors[i], markerfacecolor="None", markeredgewidth=2., markersize=8, alpha=1.0)
      if 'Azimuthal Fit Size' not in legend_dict.keys():
        legend_dict['Azimuthal Fit Size'] = plot1

  #******* T/TF vs. N FUGACITY
  xT = numpy.array( [number.nominal_value] )
  yT = numpy.array( [TF_az_f.nominal_value] )
  xTerr = numpy.array( [number.std_dev()] )
  yTerr = numpy.array( [TF_az_f.std_dev()] )
  if yTerr[0] < maxerror* yT[0] and evapimage > 3.0:
    axTFN.errorbar(xT,yT, xerr=xTerr, yerr=yTerr, fmt='D', ecolor=colors[i], markeredgecolor=colors[i], markerfacecolor="None", markeredgewidth=2., markersize=8, alpha=1.0)

  #******* T/TF vs. N SIZE
  xT = numpy.array( [number.nominal_value] )
  yT = numpy.array( [TF_az_s.nominal_value] )
  xTerr = numpy.array( [number.std_dev()] )
  yTerr = numpy.array( [TF_az_s.std_dev()] )
  if yTerr[0] < maxerror* yT[0]:
    axTFN.errorbar(xT,yT, xerr=xTerr, yerr=yTerr, fmt='x', ecolor=colors[i], markeredgecolor=colors[i], markerfacecolor="None", markeredgewidth=2., markersize=8, alpha=1.0)

  return TF_az_f, TF_az_s, T_az_s

#--------------------------------------------------
#   TEMPERATURES FROM 2D FITS
#
#   From the polylog fit to the column density
#   data one can extract T/TF directly from the 
#   fugacity or indirectly through determination
#   of T from the size+trapfreqs.  In this case
#   there are independent T's from axial and 
#   radial sizes.  The fermi temperature is 
#   opbained from the number and trap freqs.
#
#   This function returns the fugacity estimate
#   of T/TF and the axial and radial estimates
#   of T
#--------------------------------------------------
def t_2d( points, tfermi, number, i , ax_tof, ax_T , ax_TF, axTFN, legend_dict, xT):

  tof = points[:,k['ODT:odttof']]
  TF_2d_fug = []   # T/TF from 2D fugacity
  T_2d_radial = [] # T from 2D radial size
  T_2d_axial = []  # T from 2D axial size
  for j,t in enumerate(tof):
    if t > 0.0 and points[j,k['CPP:TF_2d']] < 1.e5: 
      TF_2d_fug.append(  points[j,k['CPP:TF_2d']] )
      T_2d_radial.append( points[j,k['CPP:T_2d_rd']]  ) 
      T_2d_axial.append( points[j,k['CPP:T_2d_ax']]  ) 
  
  TF_2d_f = unc.ufloat( (numpy.mean( TF_2d_fug), numpy.std( TF_2d_fug ) ) )
  T_2d_r = unc.ufloat( (numpy.mean( T_2d_radial), numpy.std( T_2d_radial ) ) ) 
  T_2d_a = unc.ufloat( (numpy.mean( T_2d_axial), numpy.std( T_2d_axial ) ) )
 
  ax_tof.plot( tof, points[:,k['CPP:TF_2d']], 's', markeredgecolor=colors[i], markerfacecolor="None", markeredgewidth=1.0, markersize=3, alpha=1.0)
  #ax_tof.plot( tof, points[:,k['CPP:T_2d_rd']]/tfermi.nominal_value, '<', markeredgecolor=colors[i], markerfacecolor="None", markeredgewidth=2., markersize=8, alpha=1.0)
  #ax_tof.plot( tof, points[:,k['CPP:T_2d_ax']]/tfermi.nominal_value, '>', markeredgecolor=colors[i], markerfacecolor="None", markeredgewidth=2., markersize=8, alpha=1.0)

  evapimage = points[0,k['EVAP:image']]/1000.

  #******* T/TF FUGACITY
  #xT = numpy.array( [ evapimage ] )
  T_2d_f = TF_2d_f * tfermi
  yT = numpy.array( [T_2d_f.nominal_value ] )
  yTerr = numpy.array( [T_2d_f.std_dev() ] ) 
  if xT[0] > 3.: # This defines the EVAP:image cuttof for T2D fugacity points
    if yTerr[0] < maxerror* yT[0]:
      for ax in ax_T:
        ax.errorbar(xT, yT, yerr=yTerr, fmt='s', ecolor=colors[i], markeredgecolor=colors[i], markerfacecolor="None", markeredgewidth=2., markersize=8, alpha=1.0)
    yT = numpy.array( [TF_2d_f.nominal_value ] )
    yTerr = numpy.array( [TF_2d_f.std_dev() ] )
    if yTerr[0] < maxerror* yT[0]:
      for ax in ax_TF:
        plot1 = ax.errorbar(xT, yT, yerr=yTerr, fmt='s', ecolor=colors[i], markeredgecolor=colors[i], markerfacecolor="None", markeredgewidth=2., markersize=8, alpha=1.0)
        if '2D Fit Fugacity' not in legend_dict.keys():
          legend_dict['2D Fit Fugacity'] = plot1

  #******* T RADIAL SIZE
  yT = numpy.array( [T_2d_r.nominal_value ] )
  yTerr = numpy.array( [T_2d_r.std_dev() ] )
  if False:
    if yTerr[0] < maxerror* yT[0]:
      for ax in ax_T:
        ax.errorbar(xT, yT, yerr=yTerr, fmt='<', ecolor=colors[i], markeredgecolor=colors[i], markerfacecolor="None", markeredgewidth=2., markersize=8, alpha=1.0)
    TF_2d_r = T_2d_r / tfermi
    yT = numpy.array( [TF_2d_r.nominal_value ] )
    yTerr = numpy.array( [TF_2d_r.std_dev() ] )
    if yTerr[0] < maxerror* yT[0]:
      for ax in ax_TF:
        ax.errorbar(xT, yT, yerr=yTerr, fmt='<', ecolor=colors[i], markeredgecolor=colors[i], markerfacecolor="None", markeredgewidth=2., markersize=8, alpha=1.0)

  #******* T AXIAL SIZE
  yT = numpy.array( [T_2d_a.nominal_value ] )
  yTerr = numpy.array( [T_2d_a.std_dev() ] )
  if False:
    if yTerr[0] < maxerror* yT[0]:
      for ax in ax_T:
        ax.errorbar(xT, yT, yerr=yTerr, fmt='>', ecolor=colors[i], markeredgecolor=colors[i], markerfacecolor="None", markeredgewidth=2., markersize=8, alpha=1.0)
    TF_2d_a = T_2d_a / tfermi
    yT = numpy.array( [TF_2d_a.nominal_value ] )
    yTerr = numpy.array( [TF_2d_a.std_dev() ] )
    if yTerr[0] < maxerror* yT[0]:
      for ax in ax_TF:
        ax.errorbar(xT, yT, yerr=yTerr, fmt='>', ecolor=colors[i], markeredgecolor=colors[i], markerfacecolor="None", markeredgewidth=2., markersize=8, alpha=1.0)

  #******* T/TF vs. N TOF FUGACITY
  xT = numpy.array( [number.nominal_value] )
  yT = numpy.array( [TF_2d_f.nominal_value] )
  xTerr = numpy.array( [number.std_dev()] )
  yTerr = numpy.array( [TF_2d_f.std_dev()] )
  if yTerr[0] < maxerror* yT[0] and evapimage > 3.0:
    axTFN.errorbar(xT,yT, xerr=xTerr, yerr=yTerr, fmt='s', ecolor=colors[i], markeredgecolor=colors[i], markerfacecolor="None", markeredgewidth=2., markersize=8, alpha=1.0)
   
  return TF_2d_f, T_2d_r, T_2d_a

#--------------------------------------------------
#   PLOT OF ETA : RATIO OF TRAP DEPTH TO TEMPERATURE
#
#   ...
#--------------------------------------------------
def eta_plot(points, depth, i, T, T_az_s, ax, xT):
  evapimage = points[0,k['EVAP:image']]/1000.

  #******* ETA AZIMUTHAL FUGACITY
  etaAZ = depth / T_az_s
  #xT = numpy.array( [ evapimage ] )
  yT = numpy.array( [ etaAZ.nominal_value ] )
  yTerr = numpy.array( [ etaAZ.std_dev() ] ) 
  if yTerr[0] < maxerror* yT[0]:
    ax.errorbar(xT, yT, yerr=yTerr, fmt='x', ecolor=colors[i], markeredgecolor=colors[i], markerfacecolor="None", markeredgewidth=1., markersize=4, alpha=1.0)


  #******* T/TF AZIMUTHAL FUGACITY
  etaAZ = depth / T
  yT = numpy.array( [ etaAZ.nominal_value ] )
  yTerr = numpy.array( [ etaAZ.std_dev() ] ) 
  if yTerr[0] < maxerror* yT[0]:
    ax.errorbar(xT, yT, yerr=yTerr, fmt='o', ecolor=colors[i], markeredgecolor=colors[i], markerfacecolor="None", markeredgewidth=1., markersize=4, alpha=1.0)

# --------------------- MAIN CODE  --------------------#


if __name__ == "__main__":
  parser = argparse.ArgumentParser('plotevap.py')
  parser.add_argument('RANGE', action="store", type=str, help='range of shots to be considered for plotevap')
  parser.add_argument('--var', action="store", help='SEC:KEY for variable that is used in x axis')
  parser.add_argument('--xlim', action="store", help='Plotting limits for x axis. Must be in the form: x0,xf ')
  parser.add_argument('--sizes', action="store",help='Set flag if you want to see the SIZE vs. TOF plot instead of T/TF vs TOF.' )

  args = parser.parse_args()
  #print os.getcwd()
  #print args.RANGE

  print args.var
  if args.var not in keys:
    parameter = 'EVAP:image'
  else:
    parameter = args.var

  print args.xlim
  if args.xlim != None:
    xlim = args.xlim.split(',')
    xlim = [ float(i) for i in xlim ]
    print xlim
  else:
    xlim = None

  print args.sizes
  sizes = None

  print "Plotting with VARIABLE = %s" % parameter

  #
  # EXTRACT DATA FROM REPORTS 
  # 
  evapdat, errmsg, raw = qrange.qrange( os.getcwd() +'/' , args.RANGE, ' '.join(keys))
  extracted = extract_same( evapdat, parameter )

  #
  # EXTRACT ODT and BFIELD TRAJECTORIES 
  #
  lowestdepth = sorted(extracted.keys())[-1]
  print lowestdepth
  lowestdepth_report = configobj.ConfigObj ( "report%04d.INI" % extracted[lowestdepth][0][k['SEQ:shot']] )
  stepsize = float( lowestdepth_report['EVAP']['evapss'] )
  maxdepth = float( lowestdepth_report['ODTCALIB']['maxdepth'] )

  lowestdepth_rampfile =  re.sub('L:', '/lab', lowestdepth_report['EVAP']['ramp'] )
  if '_phys' != lowestdepth_rampfile[-5:]:
    print "Appending _phys to the ramp file obtained from the report"
    lowestdepth_rampfile = lowestdepth_rampfile + '_phys'
  evapramp = numpy.fromfile( lowestdepth_rampfile , sep='\n')
  evapramp = evapramp / 10. * maxdepth / 5.  # Trap depth divided by 5

  try:
    lowestdepth_fieldfile =  re.sub('L:', '/lab', lowestdepth_report['EVAP']['ramp_field'] )
    fieldramp = numpy.fromfile( lowestdepth_fieldfile , sep=',')
    fieldramp = fieldramp * 6.8 * 18.06 # Magnetic field in gauss  6.8 G/A, and 18.06 A/Volt
  except:
    fieldramp = None

  evaptime = numpy.linspace( 0.0, evapramp.shape[0]*stepsize/1000., evapramp.shape[0] )

  # 
  # SETUP MATPLOTLIB
  #
  matplotlib.rcdefaults()
  latex = False 
  if latex: 
    matplotlib.rc('text',usetex=True)
    matplotlib.rcParams['text.latex.preamble']=[r"\usepackage[lf,mathtabular]{MyriadPro}",r'\usepackage{mdsymbol}']
    iflatex = r'\figureversion{lf,tab}'
  else: 
    iflatex = ''

  figscale = 1.2
  figw = 12.0 * figscale
  figh = 8.5 * figscale
  fig = plt.figure( figsize=(figw,figh) )

  # TRAJECTORY AXES
  axTRAJ = fig.add_axes( [0.05,0.08,0.28,0.28])
  axTRAJ.plot( evaptime, evapramp)
  if fieldramp != None:
    print "evaptime:", evaptime.shape
    print "fieldramp:", fieldramp.shape
    axTRAJ2 = axTRAJ.twinx()
    try:
      axTRAJ2.plot( evaptime, fieldramp[0:-1])
    except:
      print "Error plotting field ramp"
   

  # TEMPERATURE AXES
  axT = fig.add_axes( [0.05,0.665,0.25,0.32])
  axTzoom = fig.add_axes( [0.05,0.44,0.25,0.22])
  axesT = [ axT, axTzoom ]

  # ETA AXES
  axETA = axT.twinx() 

  # T/T_FERMI AXES
  axTF = fig.add_axes( [0.365,0.665,0.25,0.32])
  axTFzoom = fig.add_axes( [0.365,0.44,0.25,0.22])
  axesTF = [ axTF, axTFzoom ]
  for aa,ax in enumerate(axesTF):
    aux = ax.twinx()
    if parameter == 'EVAP:image':
      aux.plot( evaptime, evapramp )
    if aa == 0:
      aux.set_ylim(0,80.)
    if aa == 1:
      aux.set_ylim(0,8.)
    for tick in aux.yaxis.get_major_ticks():
      tick.set_visible(False)
 
  # NUMBER AND DENSITY AXES
  axN = fig.add_axes( [0.69,0.665,0.25,0.32])
  axNzoom = fig.add_axes( [0.69,0.44,0.25,0.22])
  axesN = [ axN, axNzoom ]
  for aa,ax in enumerate(axesN):
    aux = ax.twinx()
    if parameter == 'EVAP:image':
      aux.plot( evaptime, evapramp )
    if aa == 0:
      aux.set_ylim(0,80.)
    if aa == 1:
      aux.set_ylim(0,8.)
    for tick in aux.yaxis.get_major_ticks():
      tick.set_visible(False)

  if sizes != None:
    # RADIAL SIZE AXES
    axSIZE = fig.add_axes( [0.42,0.06,0.20,0.18])
    # T/T_FERMI vs. TIME-OF-FLIGHT AXES
    axTFtof = axSIZE.twinx()
  else:
    # RADIAL SIZE AXES
    axTFtof = fig.add_axes( [0.42,0.06,0.20,0.18])
    # T/T_FERMI vs. TIME-OF-FLIGHT AXES
    axSIZE = axTFtof.twinx()

  # T/T_FERMI vs. NUMBER AXES
  axTFN = fig.add_axes( [0.69,0.08,0.25,0.28])


  #
  # ITERATE OVER VALUES OF PLOT PARAMETER 
  #
  alldat=[]
  allerr=[]
  legend_dict={}
  for i,value in enumerate( reversed(sorted(extracted.keys()))  ):
    print i, value
    #pprint.pprint( extracted[depth] )
    points = numpy.array(extracted[value])
   
    # numpy array with value, read for matplotlib plot
    xT = numpy.array ( [ value ] ) 
 
    # trap depth 
    fcpow = points[0,k['EVAP:finalcpow']]
    U0    = points[0,k['ODTCALIB:maxdepth']]
    depth = fcpow / 10. * U0
    shots  = points[:,k['SEQ:shot']] 
    print "shots:", shots
    evapimage =  numpy.array( [points[0,k['EVAP:image']]/1000. ] )[0]
    tfermi, Num, Den = t_fermi( points , axesN, xT)
    print "EVAP:image   = ", evapimage
    print "Trap Depth   = ", depth
    print "Number/1e5   = ", Num
    print "T_Fermi      = ", tfermi
    print "Density/1e12 = ", Den

    xArray = points[0,k[parameter]]
    if parameter == 'EVAP:image':
      xArray = xArray / 1000.
    if parameter == 'FESHBACH:bias':
      xArray = xArray *6.8
    if parameter == 'EVAP:fieldrampfinal':
      xArray = xArray *6.8
    xT = numpy.array( [ xArray ] )
 
    number = points[:,k['CPP:nfit']]
    T, TF, TfitX, TfitY = tof_temperature(points, tfermi, Num, i, axSIZE, axesT, axesTF, axTFN, legend_dict, xT)
    print "T  from TOF = " , T 
    print "TF from TOF = " , TF

    TF_az_f, TF_az_s, T_az_s = t_azimuthal( points, tfermi, Num, i, axTFtof, axesT, axesTF, axTFN, legend_dict, xT)
    print "T/TF Azimuthal Fugacity = ", TF_az_f
    print "T/TF Azimuthal Size     = ", TF_az_s
    print "T    Azimuthal Size     = ", T_az_s

    TF_2d_f, T_2d_r, T_2d_a = t_2d( points, tfermi, Num, i, axTFtof, axesT, axesTF,  axTFN, legend_dict, xT ) 
    print "T/TF 2D Fugacity    = ", TF_2d_f
    print "T    2D Radial Size = ", T_2d_r
    print "T    2D Axial  Size = ", T_2d_a 

    print ""
    eta_plot( points,  depth, i, T, T_az_s, axETA, xT)



    alldat.append( [ evapimage, depth, tfermi.nominal_value, Num.nominal_value, Den.nominal_value, \
                     T.nominal_value, T_az_s.nominal_value, T_2d_r.nominal_value, T_2d_a.nominal_value, \
                     TF.nominal_value, TF_az_s.nominal_value, TF_az_f.nominal_value, TF_2d_f.nominal_value ] ) 
    allerr.append( [ evapimage, depth, tfermi.std_dev(), Num.std_dev(), Den.std_dev(), \
                     T.std_dev(), T_az_s.std_dev(), T_2d_r.std_dev(), T_2d_a.std_dev(), \
                     TF.std_dev(), TF_az_s.std_dev(), TF_az_f.std_dev(), TF_2d_f.std_dev() ] )
  
#--------------------------------------------------
#   PLOT LEGEND
#--------------------------------------------------
  legend_list =[]
  label_list =[]
  for key in sorted(legend_dict.keys()):
    #print key
    for art in legend_dict[key]:
      #print art
      if hasattr( art, '__iter__'):
        for art2 in art: 
          #print "\t", art2
          matplotlib.artist.setp( art2, color='black')
          art2.markeredgecolor = 'black'
          #print matplotlib.artist.getp( art2 )
      else:
        matplotlib.artist.setp( art, color='black' , markeredgecolor='black')
        #print matplotlib.artist.getp( art )
    legend_list.append( legend_dict[key] )
    label_list.append( key )

  plt.legend( legend_list, label_list, loc = 'upper right', bbox_to_anchor = (-0.3, 1.1), numpoints = 1,  ) 
  

#--------------------------------------------------
#   TEXT FILE OUTPUT
#--------------------------------------------------
 
  header = ""
  for stri in ['#  image','depth','TFermi','Num','Dens','Ttof','T_az_s','T2d_r','T2d_a','TFtof','TFaz_s','TFaz_f','TF2d_f']:
    header = header + "%8s " % stri
 
  numpy.savetxt("evap.dat", numpy.array(alldat) , fmt='%8.2f', delimiter=' ')
  outfile = open("evap.dat","r")
  outstring = outfile.read()
  outfile.close()
  outfile = open("evap.dat","w")
  outfile.write(header + '\n' + outstring)
  outfile.close()
  
#--------------------------------------------------
#   ETA PLOT
#--------------------------------------------------
  axT.yaxis.set_major_formatter( matplotlib.ticker.FormatStrFormatter(r'%d'))

  fsize = 12
  for tick in axETA.yaxis.get_major_ticks():
    tick.label.set_fontsize(fsize)

  axETA.set_ylim(0.,10.)
  axETA.set_ylabel(r"$\eta$", fontsize=fsize, va='top', labelpad=-30)

#--------------------------------------------------
#   PLOT PARAMETER LABEL
#--------------------------------------------------
  if parameter == 'EVAP:image':
    plotxlabel = r"Evaporation time (s)"
  elif parameter == 'FESHBACH:bias':
    plotxlabel = r"Bfield (G)"
  elif parameter == 'EVAP:fieldrampfinal':
    plotxlabel = r"Final Bfield (G)"
  else:
    plotxlabel = parameter

  print "plotxlabel = %s" % plotxlabel

#--------------------------------------------------
#   TRAJECTORY PLOT
#--------------------------------------------------
  axTRAJ.spines["bottom"].set_linewidth(2)
  axTRAJ.spines["top"].set_linewidth(2)
  axTRAJ.spines["left"].set_linewidth(2)
  axTRAJ.spines["right"].set_linewidth(2)

  axTRAJ.set_xlabel( r"Evaporation time (s)", fontsize=fsize, labelpad=8)
  axTRAJ.set_ylabel(r"$\mathrm{U(t)/5}\ (\mu \mathrm{K})$", fontsize=fsize, labelpad=6)
  if fieldramp != None:
    axTRAJ2.set_ylabel(r"$\mathrm{B-field}\ (\mathrm{G})$", fontsize=fsize, labelpad=6)

#--------------------------------------------------
#   TEMPERATURE PLOT
#--------------------------------------------------
  axT.grid(True)
  #axT.legend(loc='upper left', bbox_to_anchor = (0.0,-0.06), prop={'size':10}, numpoints=1)
  axT.yaxis.set_major_formatter( matplotlib.ticker.FormatStrFormatter(r'%d'))
  axT.xaxis.set_major_formatter( matplotlib.ticker.FormatStrFormatter(r'%d'))

  fsize = 12
  for tick in axT.xaxis.get_major_ticks():
    tick.label.set_fontsize(fsize)
    tick.label.set_visible(False)
  for tick in axT.yaxis.get_major_ticks():
    tick.label.set_fontsize(fsize)

  axT.set_ylim(0.,80.)
  if xlim != None:
    axT.set_xlim(xlim[0],xlim[1])
    axTzoom.set_xlim(xlim[0],xlim[1])

  axT.spines["bottom"].set_linewidth(2)
  axT.spines["top"].set_linewidth(2)
  axT.spines["left"].set_linewidth(2)
  axT.spines["right"].set_linewidth(2)

  axT.set_ylabel(r"$\mathrm{Temperature}\ (\mu \mathrm{K})$", fontsize=fsize, labelpad=6)
  #axTb.set_ylabel('Delta Position on Camera\nwith respect to red (um)', fontsize=fsize, labelpad=25, ha = 'center')

  axTzoom.grid(True)

  # Set limits of temperature zoom axis
  zoomYlim = min( 8.0 , axTzoom.get_ylim()[1])
  axTzoom.set_ylim(0., zoomYlim )
  for ax in axesT:
    if parameter == 'EVAP:image':
      ax.plot( evaptime,  evapramp, color='blue' )
  zoomYticks = numpy.linspace(0., zoomYlim, 6) 
  axTzoom.yaxis.set_major_locator( matplotlib.ticker.FixedLocator( zoomYticks[0:-1] ))

  axTzoom.spines["bottom"].set_linewidth(2)
  axTzoom.spines["top"].set_linewidth(2)
  axTzoom.spines["left"].set_linewidth(2)
  axTzoom.spines["right"].set_linewidth(2)
  axTzoom.set_xlabel(plotxlabel, fontsize=fsize, labelpad=8)
  axTzoom.set_ylabel(r"$\mathrm{Temperature} \ (\mu \mathrm{K})$", fontsize=fsize, labelpad=6)

  for tick in axTzoom.xaxis.get_major_ticks():
    tick.label.set_fontsize(fsize)
  for tick in axTzoom.yaxis.get_major_ticks():
    tick.label.set_fontsize(fsize)

#--------------------------------------------------
#   T/T_FERMI  PLOT
#--------------------------------------------------
  axTF.grid(True)
  #axTF.legend(loc='upper left', bbox_to_anchor = (0.0,-0.06), prop={'size':10}, numpoints=1)
  axTF.yaxis.set_major_formatter( matplotlib.ticker.FormatStrFormatter(r'%.2f'))
  axTF.xaxis.set_major_formatter( matplotlib.ticker.FormatStrFormatter(r'%d'))

  fsize = 12
  for tick in axTF.xaxis.get_major_ticks():
    tick.label.set_fontsize(fsize)
    tick.label.set_visible(False)
  for tick in axTF.yaxis.get_major_ticks():
    tick.label.set_fontsize(fsize)

  axTF.set_ylim(0.,2.)
  if xlim != None:
    axTF.set_xlim(xlim[0],xlim[1])
    axTFzoom.set_xlim(xlim[0],xlim[1])

  axTF.spines["bottom"].set_linewidth(2)
  axTF.spines["top"].set_linewidth(2)
  axTF.spines["left"].set_linewidth(2)
  axTF.spines["right"].set_linewidth(2)

  axTF.set_ylabel(r"$T/T_{F}$", fontsize=fsize, labelpad=6)
  #axTFb.set_ylabel('Delta Position on Camera\nwith respect to red (um)', fontsize=fsize, labelpad=25, ha = 'center')

  axTFzoom.grid(True)
  # Set limits of T/TF zoom axis
  zoomTFYlim = min( 0.8 , axTFzoom.get_ylim()[1])
  axTFzoom.set_ylim(0., zoomTFYlim )
  zoomTFYticks = numpy.linspace(0., zoomTFYlim, 6) 
  axTFzoom.yaxis.set_major_locator( matplotlib.ticker.FixedLocator( zoomTFYticks[0:-1] ))


  axTFzoom.yaxis.set_major_formatter( matplotlib.ticker.FormatStrFormatter(r'%.2f'))
  axTFzoom.spines["bottom"].set_linewidth(2)
  axTFzoom.spines["top"].set_linewidth(2)
  axTFzoom.spines["left"].set_linewidth(2)
  axTFzoom.spines["right"].set_linewidth(2)
  axTFzoom.set_xlabel(plotxlabel, fontsize=fsize, labelpad=8)
  axTFzoom.set_ylabel(r"$T/T_{F}$", fontsize=fsize, labelpad=6)

  for tick in axTFzoom.xaxis.get_major_ticks():
    tick.label.set_fontsize(fsize)
  for tick in axTFzoom.yaxis.get_major_ticks():
    tick.label.set_fontsize(fsize)

#--------------------------------------------------
#   NUMBER  PLOT
#--------------------------------------------------
  axN.grid(True)
  #axN.legend(loc='upper left', bbox_to_anchor = (0.0,-0.06), prop={'size':10}, numpoints=1)
  axN.yaxis.set_major_formatter( matplotlib.ticker.FormatStrFormatter(r'%d'))
  axN.xaxis.set_major_formatter( matplotlib.ticker.FormatStrFormatter(r'%d'))

  fsize = 12
  for tick in axN.xaxis.get_major_ticks():
    tick.label.set_fontsize(fsize)
    tick.label.set_visible(False)
  for tick in axN.yaxis.get_major_ticks():
    tick.label.set_fontsize(fsize)

  #axN.set_ylim(0.,3.)
  #axN.set_xlim(0,10)
  if xlim != None:
    axN.set_xlim(xlim[0],xlim[1])
    axNzoom.set_xlim(xlim[0],xlim[1])

  axN.spines["bottom"].set_linewidth(2)
  axN.spines["top"].set_linewidth(2)
  axN.spines["left"].set_linewidth(2)
  axN.spines["right"].set_linewidth(2)

  axN.set_ylabel(r"$\mathrm{Number}\ /10^{5}$", fontsize=fsize, labelpad=6)
  #axNb.set_ylabel('Delta Position on Camera\nwith respect to red (um)', fontsize=fsize, labelpad=25, ha = 'center')

  #axNzoom.yaxis.set_major_locator( matplotlib.ticker.FixedLocator( [ 0., .2, .4, .6]))
  axNzoom.grid(True)
  axNzoom.yaxis.set_major_formatter( matplotlib.ticker.FormatStrFormatter(r'%.1f'))
  axNzoom.spines["bottom"].set_linewidth(2)
  axNzoom.spines["top"].set_linewidth(2)
  axNzoom.spines["left"].set_linewidth(2)
  axNzoom.spines["right"].set_linewidth(2)
  axNzoom.set_xlabel(plotxlabel, fontsize=fsize, labelpad=8)
  axNzoom.set_ylabel(r"$\mathrm{Density} /10^{12} (\mathrm{cm}^{-3})$", fontsize=fsize, labelpad=6)
  
  axNzoom.yaxis.get_major_ticks()[-1].label.set_visible(False)

  for tick in axNzoom.xaxis.get_major_ticks():
    tick.label.set_fontsize(fsize)
  for tick in axNzoom.yaxis.get_major_ticks():
    tick.label.set_fontsize(fsize)

#--------------------------------------------------
#   RADIAL SIZE PLOT
#--------------------------------------------------
  axSIZE.grid(True)
  axSIZE.set_xlabel(r"Time-of-flight (ms)", fontsize=fsize/1.0, labelpad=10)
  axSIZE.set_ylabel('Radial size (um)', fontsize=fsize/1.0, labelpad=12, ha = 'center')
  axSIZE.xaxis.set_major_formatter( matplotlib.ticker.FormatStrFormatter(r'%d'))
  axSIZE.xaxis.set_major_locator( matplotlib.ticker.MultipleLocator( 1.0) )
  fsize = 12
  for tick in axSIZE.xaxis.get_major_ticks():
    tick.label.set_fontsize(fsize)
  for tick in axSIZE.yaxis.get_major_ticks():
    tick.label.set_fontsize(fsize)
  axSIZE.spines["bottom"].set_linewidth(2)
  axSIZE.spines["top"].set_linewidth(2)
  axSIZE.spines["left"].set_linewidth(2)
  axSIZE.spines["right"].set_linewidth(2)
  axSIZE.set_xlim(0,0.8) 
  axSIZE.set_ylim(0,75)
  axSIZE.set_visible(False)
  if sizes != None:
    axSIZE.set_visible(True)
   

#--------------------------------------------------
#   T/TF vs TIME-OF-FLIGHT PLOT
#--------------------------------------------------
  axTFtof.grid(True)
  TFtofylim = min(0.25, axTFtof.get_ylim()[1])
  axTFtof.set_ylim(0,TFtofylim)
  axTFtof.set_xlabel(r"Time-of-flight (ms)", fontsize=fsize/1.0, labelpad=10)
  axTFtof.set_ylabel(r'$T/T_{F}$', fontsize=fsize/1.0, labelpad=10, ha = 'center')
  axSIZE.xaxis.set_major_formatter( matplotlib.ticker.FormatStrFormatter(r'%.1f'))
  axSIZE.xaxis.set_major_locator( matplotlib.ticker.MultipleLocator( 0.2) )
  for tick in axTFtof.xaxis.get_major_ticks():
    tick.label.set_fontsize(fsize)
  for tick in axTFtof.yaxis.get_major_ticks():
    tick.label.set_fontsize(fsize)
  axTFtof.spines["bottom"].set_linewidth(2)
  axTFtof.spines["top"].set_linewidth(2)
  axTFtof.spines["left"].set_linewidth(2)
  axTFtof.spines["right"].set_linewidth(2)
  if sizes != None:
    axTFtof.set_visible(False)

#--------------------------------------------------
#   T/TF vs NUMBER PLOT
#--------------------------------------------------
  axTFN.grid(True)
  axTFN.set_xlim(0.,50.)
  TFYlim = min( 0.6 , axTFN.get_ylim()[1])
  axTFN.set_ylim(0,TFYlim)
  axTFN.set_xlabel(r"$\mathrm{Number}\ /10^{5}$", fontsize=fsize/1.0, labelpad=10)
  axTFN.set_ylabel(r'$T/T_{F}$', fontsize=fsize/1.0, labelpad=10, ha = 'center')
  for tick in axTFN.xaxis.get_major_ticks():
    tick.label.set_fontsize(fsize)
  for tick in axTFN.yaxis.get_major_ticks():
    tick.label.set_fontsize(fsize)
  axTFN.spines["bottom"].set_linewidth(2)
  axTFN.spines["top"].set_linewidth(2)
  axTFN.spines["left"].set_linewidth(2)
  axTFN.spines["right"].set_linewidth(2)

  
#--------------------------------------------------
#   SAVE TO PNG
#--------------------------------------------------
  output = args.RANGE
  output = output.replace('-','m')
  output = output.replace(':','-')
  output = output.replace(',','_')
  output = "plotevap" + output + ".png"
  print output

  datestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
  #print output
  stamp = output + " plotted on " + datestamp
  fig.text( 0.01, 0.0, stamp, size=8)
  #fig.savefig( "debug.png" , dpi=140)
  fig.savefig( output , dpi=140)

