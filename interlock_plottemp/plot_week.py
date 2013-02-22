#!/usr/bin/python

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import matplotlib.dates as mdates


import datetime

import sys

from configobj import ConfigObj


rooms={'BH.B02.RT.B':'EMT2','BH.B02.RT.C':'EMT1','BH.B02.RT.D':'APP3','BH.FCU01.SPT':'EMT2-mini','BH.FCU02.SPT':'EMT1-mini','BH.FCU03.SPT':'APP3-mini'}




def plotreport(file,showplot = 1):
	
	try:
		print file 
		report = open(file,"r")
	except:
	   print "ERROR: Can't open Insight Trend Interval Report file"
	
	npts=0
	labels=[]
	for line in report:
		if line.startswith("\"Point"):
			pointname = line[line.find("\"",10)+1:line.find("\"",12)]
			if pointname in rooms.keys():
				npts = npts+1
				labels.append(pointname)
	#      labels.append(line[line.find("\"",10)+1:line.find("\"",12)])

	print labels

	report.close()


	try:
	   report = open(file,"r")
	except:
	   print "ERROR: Can't open Insight Trend Interval Report file"

	times=[]
	data=[]
	for line in report:
		try:
			month=int(line[1:line.index("/")] )
			sub=line[line.index("/")+1:]
			day=int(sub[0:sub.index("/")])
			sub=sub[sub.index("/")+1:]
			year=int(sub[0:sub.index("\"")])
			sub=sub[sub.index("\"")+1:]
			time=sub[sub.index("\"")+1:sub.index("\"")+9]
			sub=sub[sub.index("\"")+10:]
			 
			hour=int(time[0:2])
			minute=int(time[3:5])
			second=int(time[6:8])

			datapoint=[]
			for i in range(npts):
				 first=sub.index("\"") 
				 sub=sub[first+1:]
				 second=sub.index("\"")
				 datapoint.append( float(sub[:second]))
				 sub=sub[second+1:]
			data.append(datapoint)
			#print "month = %s, day = %s, year = %s, time = %s  %s" % (month,day,year,time,sub)
			#times.append("%02s-%02s-%02s-%s" % (month,day,year,time))
			times.append( datetime.datetime( year, month, day, hour, minute, second) )
		except:
		  continue
		  

	report.close()


	D = np.array( data )

	fig=plt.figure(figsize=(12,4))
	ax=fig.add_axes([0.05,0.3,0.75,0.6])

	leg=[]
	for i in range(npts):
	   print rooms[labels[i]]
	   leg.append(rooms[labels[i]])
	   ax.plot(times,D[:,i],label=rooms[labels[i]])

	plt.legend(leg, loc='upper right', bbox_to_anchor=(1.27,0.85))

	ax.xaxis.set_major_locator( mdates.DayLocator() ) #major ticks every day
	ax.xaxis.set_minor_locator( mdates.HourLocator(interval=4) ) #minor ticks every four hours
	ax.xaxis.set_major_formatter( mdates.DateFormatter('%Y-%m-%d %H:%M') )
	#Comment to see hour minute ticks
	#~ ax.xaxis.set_minor_formatter( mdates.DateFormatter('%H:%M') )

	fig.autofmt_xdate()


	plt.savefig(file+'.png')
	if (showplot==1):
		plt.show()
	#plt.colorbar()
	#plt.show()

if __name__ == "__main__":
	if not (len(sys.argv) == 2 ):
		print "  plot_week.py:"
		print ""
		print "  Prints the weekly temperatures in the given Insight Trend Interval Report."
		print ""
		print "  usage:  plot_week.py [REPORT]"  
		exit(1)

	 

	plotreport(sys.argv[1])
