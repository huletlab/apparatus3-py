#!/usr/bin/python
import ConfigParser
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import operator
import time
import datetime as dt
import argparse


parser = argparse.ArgumentParser("plot_webcam.py")
parser.add_argument("-D",action="store",default = 0.0,type=float,help="Plot data in X days. 0 = all data")
args = parser.parse_args()

if args.D ==0:
	print "Plotting all data. Use -D to specify how many days you want to plot. * It can be a fraction of day."
	datelimit = 0
else:
	print "Plotting data for",args.D,"days."
	datelimit = time.time()-60*60*24.0*args.D

ZOOM = 1

# convert time in the form of Y/M/D_Hms to time struct
def converttime(s):
	return time.mktime(dt.datetime.strptime(s, "%Y/%m/%d_%H%M%S").timetuple())

def findclosest( targetlist,number):
	templist = [ abs(i-number) for i in targetlist]
	cloindex, clodat = min(enumerate(templist),key = operator.itemgetter(1))
	return cloindex,targetlist[cloindex]

config = ConfigParser.RawConfigParser()
inifile = '/lab/software/apparatus3_local/utilities/Lattice_Webcam/log.history'
config.read(inifile)
Beams = config.sections()
pairs = [("GR1","IR1"),("GR2","IR2"),("GR3","IR3")]

fig = plt.figure()
fig2 = plt.figure()
for GR,IR in pairs:
	if GR == "GR1":
		ax = fig2.add_subplot(231)
		ax2 = fig2.add_subplot(234)
		ax3 = fig.add_subplot(231)
		ax4 = fig.add_subplot(234)
	elif GR == "GR2":
		ax = fig2.add_subplot(232)
		ax2 = fig2.add_subplot(235)
		ax3 = fig.add_subplot(232)
		ax4 = fig.add_subplot(235)
	elif GR == "GR3":
		ax = fig2.add_subplot(233)
		ax2 = fig2.add_subplot(236)
		ax3 = fig.add_subplot(233)
		ax4 = fig.add_subplot(236)
	else:
		continue
	
	ax.set_title(GR+"-"+IR)
	ax3.set_title(GR)
	ax4.set_title(IR)
	if ZOOM == 0:
		ax3.set_xlim(0,1064)
		ax4.set_xlim(0,1064)
		ax3.set_ylim(768,0)
		ax4.set_ylim(768,0)
	dataGR = config.items(GR)
	dataIR = config.items(IR)
	timeGR = []
	timeIR = []
	posxGR = []
	posxIR = []
	posyGR =[]
	posyIR =[]
	posdateGR= []
	posdateIR= []
	
	for i in dataGR:
		#date = int(i[0].replace("/","").replace("_",""))
		date = converttime(i[0])
		if date > datelimit:
			posdateGR.append(date)
			posGR = i[1].replace("\"","").split(",")
			posxGR.append(float(posGR[0]))
			posyGR.append(float(posGR[1]))
	colorGR = []
	
	for i,date in enumerate(posdateGR):
		colorGR.append(1.0*i/len(posdateGR))

	for i in dataIR:
		#date = int(i[0].replace("/","").replace("_",""))
		date = converttime(i[0])
		if date > datelimit:
			posIR = i[1].replace("\"","").split(",")
			posxIR.append(float(posIR[0]))
			posyIR.append(float(posIR[1]))
			posdateIR.append(date)
	colorIR = []
	
	for i,date in enumerate(posdateIR):
		colorIR.append(1.0*i/len(posdateIR))
	ax3.scatter(posxGR,posyGR,c=colorGR,cmap=plt.cm.BuGn,s = 50)
	ax3.set_xlabel("Webcam X")
	ax3.set_ylabel("Webcam Y")
	for l in ax3.get_xticklabels():
		l.set_rotation(30)
	ax4.scatter(posxIR,posyIR,c=colorIR,cmap=plt.cm.Reds,s = 50)
	ax4.set_xlabel("Webcam X")
	ax4.set_ylabel("Webcam Y") 
	for l in ax4.get_xticklabels():
		l.set_rotation(30)
	#Find nearest date and pair
	dateDelta = []
	posxDelta = []
	posyDelta = []

	for dGr,xGr,yGr in zip(posdateGR,posxGR,posyGR):
		ind,date = findclosest(posdateIR,dGr)
		if ( date-dGr <60.0):
			dateDelta.append(date)	
			posxDelta.append(xGr- posxIR[ind])
			posyDelta.append(yGr- posyIR[ind])



	color = []
	
	for i,date in enumerate(dateDelta):
		color.append(1.0*i/len(dateDelta))

	ax.scatter(posxDelta,posyDelta,c=color,cmap=plt.cm.BuPu,s = 50)
	ax.set_xlabel("GR-IR X")
	ax.set_ylabel("GR-IR Y")
	for l in ax.get_xticklabels():
		l.set_rotation(30)
	
	date = [dt.datetime.fromtimestamp(ts) for ts in dateDelta]
	ax22 = ax2.twinx()	
	ax22.plot(date,posyDelta,color = "blue")
	# Weird issue here on atomcool ax22 need to plot first to prevent destorying the DateFormatter
	ax2.plot(date,posxDelta,color="red")
	ax2.xaxis.set_major_formatter(mdates.DateFormatter('%y/%m/%d %H:%M'))
	ax2.set_ylabel("GR-IR X",color = "red")
	ax2.set_xlabel("Time")
	ax22.set_ylabel("GR-IR Y",color="blue")
	for l in ax2.get_xticklabels():
		l.set_rotation(30)

fig.tight_layout()
fig2.tight_layout()
plt.show()
