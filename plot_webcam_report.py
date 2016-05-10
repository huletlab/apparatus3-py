#!/usr/bin/python
import ConfigParser
import matplotlib.pyplot as plt
import matplotlib
import operator
import time
import datetime as dt
import argparse
import qrange 
import os

parser = argparse.ArgumentParser("plot_webcam.py")
parser.add_argument('--range', action = "store", \
         help="range of shots to be used.")
parser.add_argument('--output', action="store", \
        help="optional path of png to save figure")

args = parser.parse_args()
rangestr = args.range.replace(':','to')
rangestr = rangestr.replace(',','_')

beams = ["GR1","GR2","GR3","IR1","IR2","IR3"]
data = {}

for beam in beams:
	keys = ["SEQ:shot"]
	for k in ["_Cx","_Cy","_Wx","_Wy"]:
		keys.append("LATTICEWEBCAM:"+beam+k)
	data[beam]  =  qrange.qrange_eval("./",args.range,keys)[0]


ZOOM = 1

pairs = [("GR1","IR1"),("GR2","IR2"),("GR3","IR3")]

fig = plt.figure(figsize=(16,12))
fig2 = plt.figure(figsize=(16,12))
gs = matplotlib.gridspec.GridSpec(3,3,wspace =0.5, hspace =0.6,top=0.9,left = 0.07,right=0.93,bottom=0.1)
gs2 = matplotlib.gridspec.GridSpec(2,3,wspace =0.5, hspace =0.5,top=0.9,left = 0.07,right=0.93,bottom=0.1)

for GR,IR in pairs:
	if GR == "GR1":
		#ax = fig2.add_subplot(231)
		#ax2 = fig2.add_subplot(234)
		#ax3 = fig.add_subplot(231)
		#ax4 = fig.add_subplot(234)
		ax = fig2.add_subplot(gs2[0,0])
		ax2 = fig2.add_subplot(gs2[1,0])
		ax3 = fig.add_subplot(gs[0,0])
		ax4 = fig.add_subplot(gs[1,0])
		ax5 = fig.add_subplot(gs[2,0])
	elif GR == "GR2":
		#ax = fig2.add_subplot(232)
		#ax2 = fig2.add_subplot(235)
		#ax3 = fig.add_subplot(232)
		#ax4 = fig.add_subplot(235)
		ax = fig2.add_subplot(gs2[0,1])
		ax2 = fig2.add_subplot(gs2[1,1])
		ax3 = fig.add_subplot(gs[0,1])
		ax4 = fig.add_subplot(gs[1,1])
		ax5 = fig.add_subplot(gs[2,1])
	elif GR == "GR3":
		#ax = fig2.add_subplot(233)
		#ax2 = fig2.add_subplot(236)
		#ax3 = fig.add_subplot(233)
		#ax4 = fig.add_subplot(236)
		ax = fig2.add_subplot(gs2[0,2])
		ax2 = fig2.add_subplot(gs2[1,2])
		ax3 = fig.add_subplot(gs[0,2])
		ax4 = fig.add_subplot(gs[1,2])
		ax5 = fig.add_subplot(gs[2,2])
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
	dataGR = data[GR]
	dataIR = data[IR]
	timeGR = []
	timeIR = []
	posxGR = []
	posxIR = []
	posyGR =[]
	posyIR =[]
	shotGR= []
	shotIR= []
	
	for i in dataGR:
		#date = int(i[0].replace("/","").replace("_",""))
		shot = i[0]
		shotGR.append(shot)
		posGR = i[1:3]
		posxGR.append(float(posGR[0]))
		posyGR.append(float(posGR[1]))
	colorGR = []
	
	for i,date in enumerate(shotGR):
		colorGR.append(1.0*i/len(shotGR))
	for i in dataIR:
		#date = int(i[0].replace("/","").replace("_",""))
		shot = i[0]
		posIR = i[1:3]
		posxIR.append(float(posIR[0]))
		posyIR.append(float(posIR[1]))
		shotIR.append(shot)
	colorIR = []
	
	for i,date in enumerate(shotIR):
		colorIR.append(1.0*i/len(shotIR))
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
	posxGRmod = [ i - posxGR[0] for i in posxGR]
	posyGRmod = [ i - posyGR[0] for i in posyGR]
	posxIRmod = [ i - posxIR[0] for i in posxIR]
	posyIRmod = [ i - posyIR[0] for i in posyIR]
	ax5.plot(shotGR,posxGRmod,c="blue",label="GRX+%.1f"%(posxGR[0]))
	ax5.plot(shotGR,posyGRmod,c="green",label="GRY+%.1f"%(posyGR[0]))
	ax5.plot(shotIR,posxIRmod,c="darkred",label="IRX+%.1f"%(posxIR[0]))
	ax5.plot(shotIR,posyIRmod,c="deeppink",label="IRY+%.1f"%(posyIR[0]))
	ax5.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,ncol=2, mode="expand", borderaxespad=0.)
	ax5.set_xlabel("Shot")
	ax5.set_ylabel("Webcam Position")
	for l in ax5.get_xticklabels():
		l.set_rotation(40) 
	#ax5.set_xticklabels(ax5.get_xticklabels(),rotation=45)
	#Find nearest date and pair
	if not( shotIR == shotGR):
		print "Shots doens't match! exit!"
		exit(1)

	shotDelta = shotIR
	posxDelta = []
	posyDelta = []

	for ind,s in enumerate(shotIR):
		posxDelta.append(posxGR[ind]- posxIR[ind])
		posyDelta.append(posyGR[ind]- posyIR[ind])



	color = []
	
	for i,date in enumerate(shotDelta):
		color.append(1.0*i/len(shotDelta))

	ax.scatter(posxDelta,posyDelta,c=color,cmap=plt.cm.BuPu,s = 50)
	ax.set_xlabel("GR-IR X")
	ax.set_ylabel("GR-IR Y")
	for l in ax.get_xticklabels():
		l.set_rotation(30)
	
	ax22 = ax2.twinx()	
	ax22.plot(shotDelta,posyDelta,color = "blue")
	# Weird issue here on atomcool ax22 need to plot first to prevent destorying the DateFormatter
	ax2.plot(shotDelta,posxDelta,color="red")
	#ax2.xaxis.set_major_formatter(mdates.DateFormatter('%y/%m/%d %H:%M'))
	ax2.set_ylabel("GR-IR X",color = "red")
	ax2.set_xlabel("Shot")
	ax22.set_ylabel("GR-IR Y",color="blue")
	for l in ax2.get_xticklabels():
		l.set_rotation(30)
	#ax2.set_xticklabels(ax2.get_xticklabels(),rotation=45)

if args.output != None:
        print args.output
        if not os.path.exists(args.output):
                os.makedirs(args.output)
        print "Saving figure to %s" % args.output
        fig.savefig( args.output+"lattice_webcam_log_"+rangestr+".png", dpi=240 )
        fig2.savefig( args.output+"lattice_webcam_log_diff_"+rangestr+".png", dpi=240 )
else:
        if not os.path.exists("plots"):
                os.makedirs("plots")
        fig.savefig( "plots/lattice_webcam_log_"+rangestr+".png", dpi=240 )
        fig2.savefig( "plots/lattice_webcam_log_diff_"+rangestr+".png", dpi=240 )


