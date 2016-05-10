#!/usr/bin/python
import queue
import argparse
import qrange
import os
import math
from configobj import ConfigObj
import matplotlib
import matplotlib.pyplot as plt
import datetime
import re
from scipy import stats
import copy

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'


class autoAlignGr:
	def __init__(self,analyze = "analyze # -f -R 100,100,300,300"):
		self.GRs = ["GR1H","GR1V","GR2H","GR2V","GR3H","GR3V"]
		self.IRs = ["1+2","2+3","1+3"]
		self.keysToRead = ("CPP:ax0w","CPP:ax1w","CPP:ax0c","CPP:ax1c")
		iniPath = os.path.abspath(__file__).replace(".py",".ini")
		self.INI = ConfigObj(iniPath)
		self.IR_PAIR_CONDITION = 1.0
		self.POS_CONDITION = 1.0
		self.MAX_ITERATION = 20.0
		self.MAX_PICO = 5.0
		self.FLAG = {}
		self.SPico = {}
		self.GR_SLOPE={}
		self.IR_GOAL = {}
		self.GR_GOAL={}
		self.GR_POS={}
		self.SHOT={}
		self.DELTA={}
		self.DeltaLOG={"X":{},"Y":{}}
		self.WebcamLOG={"GR":{"CX":{},"CY":{},"WX":{},"WY":{}},"IR":{"CX":{},"CY":{},"WX":{},"WY":{}}}
		self.WebcamPLOT={}
		self.PicoLOG={}
		self.ANALYSIS = analyze
		self.SAVEFOLDER = "autoAlign/"+datetime.datetime.now().strftime("%H_%M/")
		self.INITSHOT = queue.shotNum()
		for green in self.GRs:
			self.FLAG[green] = 0
			self.GR_SLOPE[green]=tuple([ float(i) for i in self.INI["SLOPE"][green]])
			self.SHOT[green]=-1
			self.DeltaLOG["X"][green]=[]
			self.DeltaLOG["Y"][green]=[]
			self.PicoLOG[green]=[]
			self.SPico[green]=0
			self.DELTA[green]=(0,0)
			
		for ir in self.IRs:
			self.SHOT[ir]=-1

		#Setup figures
		self.AX ={}
		self.AXm ={}
		self.FIGURE = plt.figure(figsize=(12, 12))
		gs = matplotlib.gridspec.GridSpec(3,3,wspace =0.3, hspace =0.4,top=0.9,left = 0.07,right=0.93,bottom=0.1)
		
		for i,green in enumerate(self.GRs):
		        if i == 0:
		                self.AX[green] = self.FIGURE.add_subplot(gs[i%2,i/2])
		                ax0 = self.AX[green]
		                self.AXm[green] = self.AX[green].twinx()
		                ax0m = self.AXm[green]
		        else:
		                # this is for sharing the scales properly
		                self.AX[green] = self.FIGURE.add_subplot(gs[i%2,i/2],sharey=ax0)
		                self.AXm[green] = self.AX[green].twinx()
		                self.AXm[green].get_shared_y_axes().join(self.AXm[green],ax0m)
	
		for beam in range(1,4):
			self.WebcamLOG["GR"]["CX"][beam] = []
			self.WebcamLOG["GR"]["CY"][beam] = []
			self.WebcamLOG["IR"]["CX"][beam] = []
			self.WebcamLOG["IR"]["CY"][beam] = []
			self.WebcamLOG["GR"]["WX"][beam] = []
			self.WebcamLOG["GR"]["WY"][beam] = []
			self.WebcamLOG["IR"]["WX"][beam] = []
			self.WebcamLOG["IR"]["WY"][beam] = []
			self.WebcamPLOT[beam] = self.FIGURE.add_subplot(gs[2,beam-1])



	def updateLog(self):
		for green in self.GRs:
			pico = self.PicoLOG[green][-1] if self.PicoLOG[green] else 0
			spico = self.SPico[green] if self.SPico[green] else 0
			self.PicoLOG[green].append(spico+pico)
			self.DeltaLOG["X"][green].append(self.DELTA[green][0])
			self.DeltaLOG["Y"][green].append(self.DELTA[green][1])

		for beam in range(1,4):
			lshot = max(self.SHOT.values())
			gcx = queue.readReport(shotnum = lshot,keys=["LATTICEWEBCAM:GR%d"%beam+"_Cx"]).values()[0]
			gcy = queue.readReport(shotnum = lshot,keys=["LATTICEWEBCAM:GR%d"%beam+"_Cy"]).values()[0]
			icx = queue.readReport(shotnum = lshot,keys=["LATTICEWEBCAM:IR%d"%beam+"_Cx"]).values()[0]
			icy = queue.readReport(shotnum = lshot,keys=["LATTICEWEBCAM:IR%d"%beam+"_Cy"]).values()[0]
			gwx = queue.readReport(shotnum = lshot,keys=["LATTICEWEBCAM:GR%d"%beam+"_Wx"]).values()[0]
			gwy = queue.readReport(shotnum = lshot,keys=["LATTICEWEBCAM:GR%d"%beam+"_Wy"]).values()[0]
			iwx = queue.readReport(shotnum = lshot,keys=["LATTICEWEBCAM:IR%d"%beam+"_Wx"]).values()[0]
			iwy = queue.readReport(shotnum = lshot,keys=["LATTICEWEBCAM:IR%d"%beam+"_Wy"]).values()[0]
			self.WebcamLOG["GR"]["CX"][beam].append(float(gcx))
			self.WebcamLOG["GR"]["CY"][beam].append(float(gcy))
			self.WebcamLOG["IR"]["CX"][beam].append(float(icx))
			self.WebcamLOG["IR"]["CY"][beam].append(float(icy))
			self.WebcamLOG["GR"]["WX"][beam].append(float(gwx))
			self.WebcamLOG["GR"]["WY"][beam].append(float(gwy))
			self.WebcamLOG["IR"]["WX"][beam].append(float(iwx))
			self.WebcamLOG["IR"]["WY"][beam].append(float(iwy))


	def plotLog(self,show = 1,save = 0,usemirror = 1):
		#plt.ioff()
		for green in self.GRs:
			self.AX[green].cla()
			self.AXm[green].cla()
       			self.AX[green].set_title(green,color="green" if self.FLAG[green] else "grey")
        		colorx = "red" if (self.GR_SLOPE[green][0] != 0 ) else "grey"
       			colory = "red" if (self.GR_SLOPE[green][1] != 0 ) else "grey"
       			if usemirror == 1:
				xaxis = range(len(self.DeltaLOG["X"][green]))
       				self.AX[green].set_xlabel("Iteration")
				l1 = self.AX[green].plot(xaxis,self.DeltaLOG["X"][green],label=green+"_X",color = colorx)
       				l2 = self.AX[green].plot(xaxis,self.DeltaLOG["Y"][green],label=green+"_Y",color = colory)
       				l3 = self.AXm[green].plot(xaxis,self.PicoLOG[green],color = "black",label="Pico")
       				self.AX[green].axhline(y=1,color = "deeppink")
       				self.AX[green].axhline(y=-1,color = "deeppink")
       				plots = l1+l2+l3
       				labels = [ l.get_label() for l in plots]

				if re.findall(r'\d+',green)[0] =="1":
       					self.AX[green].set_ylabel("Delta Position(pixel)")
				if re.findall(r'\d+',green)[0] =="3":
       					self.AXm[green].set_ylabel("Pico Movement")

				self.AX[green].legend(plots, labels, loc=0)
			else:	
				self.AXm[green].axes.get_yaxis().set_visible(False)
				xaxis = self.PicoLOG[green]
       				self.AX[green].set_xlabel("Pico Movement")
				l1 = self.AX[green].plot(xaxis,self.DeltaLOG["X"][green],label=green+"_X",color = colorx)
       				l2 = self.AX[green].plot(xaxis,self.DeltaLOG["Y"][green],label=green+"_Y",color = colory)
       				plots = l1+l2
				##Fit to a line
				if len(self.DeltaLOG["Y"][green]) > 3:
					slopeInINI = self.GR_SLOPE[green][0] if (self.GR_SLOPE[green][0] != 0 ) else self.GR_SLOPE[green][1]
					yaxis = self.DeltaLOG["X"][green] if (self.GR_SLOPE[green][0] != 0 ) else self.DeltaLOG["Y"][green]
					slope,inter,r,p,serr = stats.linregress(xaxis,yaxis)
					fitline = [slope * x +inter for x in xaxis ]
					l3 = self.AX[green].plot(xaxis,fitline,label="Fit:1/m:%.2f;INI:%.2f" %(1.0/slope,slopeInINI),color = "blue")
       					plots = l1+l2+l3
       				self.AX[green].axhline(y=1,color = "grey")
       				self.AX[green].axhline(y=-1,color = "grey")
				if re.findall(r'\d+',green)[0] =="1":
       					self.AX[green].set_ylabel("Delta Position(pixel)")
       				labels = [ l.get_label() for l in plots]
				self.AX[green].legend(plots, labels, loc=0,prop={"size":10})
	

		plotrange = 100
		for beam in range(1,4):
			pl = self.WebcamPLOT[beam]
			pl.cla()
			pl.set_title("Webcam Beam %d" %beam)
			pl.set_xlabel("Webcam X")
			pl.set_ylabel("Webcam Y")
			grx = self.WebcamLOG["GR"]["CX"][beam]
			gry = self.WebcamLOG["GR"]["CY"][beam]
			irx = self.WebcamLOG["IR"]["CX"][beam]
			iry = self.WebcamLOG["IR"]["CY"][beam]
			grwx = self.WebcamLOG["GR"]["WX"][beam]
			grwy = self.WebcamLOG["GR"]["WY"][beam]
			irwx = self.WebcamLOG["IR"]["WX"][beam]
			irwy = self.WebcamLOG["IR"]["WY"][beam]
			pl.plot(grx,gry,"-o",ms=1,mfc="green",mec="green",color="green")
			pl.plot(irx,iry,"-o",ms=1,mfc="red",mec="red",color="red")
			for i, (gx,gy,ix,iy,gwx,gwy,iwx,iwy) in enumerate(zip(grx,gry,irx,iry,grwx,grwy,irwx,irwy)):
				alp = (i+1.0)/(len(grx))*0.9
				gcircle = matplotlib.patches.Ellipse((gx,gy),width = gwx,height= gwy,angle=0,color = "green",alpha = alp)
				icircle = matplotlib.patches.Ellipse((ix,iy),width = iwx,height= iwy,angle=0,color= "deeppink",alpha = alp)
				pl.add_artist(gcircle)
				pl.add_artist(icircle)
			pl.text(grx[-1]+10,gry[-1]-10,"C:(%.1f,%.1f)\nW:(%.1f,%.1f)"%(grx[-1],gry[-1],grwx[-1],grwy[-1]),color="k")
			pl.text(irx[-1]+10,iry[-1],"C:(%.1f,%.1f)\nW:(%.1f,%.1f)"%(irx[-1],iry[-1],irwx[-1],irwy[-1]),color="firebrick")
			#Get the maxium ranges
			xr = pl.get_xlim()[1]-pl.get_xlim()[0]
			yr = pl.get_ylim()[1]-pl.get_ylim()[0]
			plotrange = max([plotrange,xr*2,yr*2])
		
		# Set ranges for the plot 
		plotrange  = plotrange +20 
		for beam in range(1,4):
			pl = self.WebcamPLOT[beam]
			grx = self.WebcamLOG["GR"]["CX"][beam]
			gry = self.WebcamLOG["GR"]["CY"][beam]
			pl.set_xlim([grx[-1]-plotrange/2,grx[-1]+plotrange/2])
			pl.set_ylim([gry[-1]-plotrange/2,gry[-1]+plotrange/2])
		if show:
			#plt.ion()
			plt.show(block=False)


		if save:
			if not os.path.exists(self.SAVEFOLDER):
			        os.makedirs(self.SAVEFOLDER)
			name = "summary.png" if usemirror else "summary2.png"
			self.FIGURE.savefig( self.SAVEFOLDER + name,dpi=100)


	def resetFlag(self):
		for green in self.GRs:
			self.FLAG[green] = 0 
	
	def overrideFlag(self,override):
		for i,green in enumerate(self.GRs):
			self.FLAG[green] = int(override[i])

	def reduceFlag(self):
		temp =1
		for f in self.FLAG.values():
			temp = temp *f
		return temp 

	def parseOverride(self,overList):
		override = ""
		overrideKey = ["DIMPLE:ir1","DIMPLE:ir2","DIMPLE:ir3","DIMPLE:gr1","DIMPLE:gr2","DIMPLE:gr3" , "DIMPLELATTICE:knob01"]
		for key,value in zip(overrideKey,overList):
			override = override + key + "\t%f\t"%(float(value)) 
		override = override +"DIMPLELATTICE:imgdet\t-110\tLATTICEWEBCAM:check\t1\n"
		return override

	def getDelta(self):
		for i,green in enumerate(self.GRs):
			pos = self.GR_POS[green][0]
			goalpos = self.GR_GOAL[green][0]
			self.DELTA[green] = (pos[0]-goalpos[0],pos[1]-goalpos[1])
		
	def getSuggestPico(self):
		for gr in self.GRs:
			self.SPico[gr]=0
		self.getDelta()
		for i,green in enumerate(self.GRs):
			sl = self.GR_SLOPE[green]
			delta = self.DELTA[green]
			for i,s in enumerate(sl):
				if s != 0:
					sp = delta[i]*s
					sp1 = math.copysign(1,sp)
					finalsp = sp1 if (int(sp) ==0) else int(sp) 
					self.SPico[green] = finalsp if (not self.FLAG[green]) else 0
					if abs(self.SPico[green]) > self.MAX_PICO:
						print "Warning! " +green +" suggest PICO %d is grater than PICO_MAX(%d) will use max value instead."%(self.SPico[green],self.MAX_PICO)
						self.SPico[green] = math.copysign(self.MAX_PICO,self.SPico[green])

		return self.parseSuggestPico()

	def parseSuggestPico(self):
		override = "PICO:move\t1\t"
		for key,pico in self.SPico.iteritems():
			if pico != 0 :
				override = override + "PICO:m" + key +"\t%d\t"%pico
		return override
	
	def checkAllGr(self,pico="",force=1,smart=1):
		gr_to_check = []
		if (not force):
			for green  in self.GRs:
				if (not self.FLAG[green]):
					gr_to_check.append(green) 				
		elif (smart):
			for green in self.GRs:
				if (green not in self.GR_POS.keys()):
					gr_to_check.append(green) 				
		else:
			gr_to_check = self.GRs
	
		for i,green in enumerate(gr_to_check):
			print "Checking " + green
			movepico = pico if (i ==0 ) else ""
			override = movepico + self.parseOverride(self.INI["OVERRIDE_GR"][green])
			shotnum,result = queue.runShotWaitAnalyzeRead(override,analyzeCommand=self.ANALYSIS, keys=self.keysToRead)
			self.GR_POS[green] = [(float(result["CPP:ax0c"]),float(result["CPP:ax1c"])),(float(result["CPP:ax0w"]),float(result["CPP:ax1w"]))]
			self.SHOT[green] = shotnum
		
	def checkCondition(self):	
		for green in self.GRs:
			g_pos = self.GR_POS[green][0]
			i_pos = self.GR_GOAL[green][0]
			d_pos = [g_pos[0]-i_pos[0],g_pos[1]-i_pos[1]]
			self.FLAG[green] = 0 if ((abs(d_pos[0]) > self.POS_CONDITION) or (abs(d_pos[1])> self.POS_CONDITION)) else 1
					
	##	Check Pair and Find Goals	##
	def checkIRs(self,smart=1):
		for ir in self.IRs:
			#This is a smart start feature if the IR_GOAL is already populate skip
			if((ir not in self.IR_GOAL.keys()) | (not smart)):
				print "Checking " + ir 
				override = self.parseOverride(self.INI["OVERRIDE_IR"][ir])
				shotnum,result = queue.runShotWaitAnalyzeRead(override,analyzeCommand=self.ANALYSIS, keys=self.keysToRead)
				self.IR_GOAL[ir] = [(float(result["CPP:ax0c"]),float(result["CPP:ax1c"])),(float(result["CPP:ax0w"]),float(result["CPP:ax1w"]))]
				self.SHOT[ir] = shotnum
			else:
				print "Skip checking " + ir 

		wc = self.IR_PAIR_CONDITION
		warning = 0
		warningstring = ""
		temp = copy.copy(self.IR_GOAL)
		for key,pair in self.IR_GOAL.iteritems():
			for akey,apair in temp.iteritems():
		        	if key != akey:
		                	d0 = abs(self.IR_GOAL[key][0][0]-self.IR_GOAL[akey][0][0])
		                	d1 = abs(self.IR_GOAL[key][0][1]-self.IR_GOAL[akey][0][1])
					if d0>wc:
		                        	warning = 1
		                        	warningstring = warningstring + key +" and " +akey+" separate in 0 by %.2f pixel.\n"%d0
		                	if d1>wc:
		                        	warning = 1
		                        	warningstring = warningstring + key +" and " +akey+" separate in 1 by %.2f pixel.\n"%d1
			temp.pop(key)
		
		if warning:
			print warningstring
			print "##########	IR pairs are not good! .		##########"
		else:
			print "##########	IR pairs are good continue with green.	##########"
		
		exitinput = raw_input("Press Enter to continue .. or enter anything to exit")
		
		if exitinput:
			exit(1)
		
		for green in self.GRs:
			self.GR_GOAL[green]=self.IR_GOAL[self.INI["GOAL"][green]]


		
	##	Cheking GR Pair		##
	def run(self,smartrange=""):
		self.smartStart(smartrange)
		self.checkIRs(smart = 1)
		overPico=""
		iteration = 0
		color = bcolors()
		#self.SPico = {}

		while ((iteration<self.MAX_ITERATION)):
			print color.OKBLUE,"\nCheck Green -- Iteration %d" %(iteration+1) ,color.ENDC
			if self.SPico:
				picoString =""
				for green in self.GRs:
					c = color.OKGREEN if (self.SPico[green]==0) else color.FAIL
					picoString  = picoString + green + " " + c+  " " + "%4d"%self.SPico[green] +" " + color.ENDC 
				print "Move Pico:\t\t",picoString
			
			exitinput = raw_input("Press Enter to continue, enter r to rest the flags, enter \"111111\" for example to overide the flags, or enter anything else to exit:")
			
			if exitinput=="r":
				print "Reseting the flags. Will check all shots again."
				self.resetFlag()
			elif (exitinput.isdigit() and len(exitinput) ==6 ):
				print "Overriding the flags. Will check corresponding shots again."
				self.overrideFlag(exitinput)
				overPico = self.getSuggestPico()
			elif exitinput:
				break
			else:
				print "Continues."
		
			
			self.checkAllGr(overPico,force=(not iteration),smart=(not iteration))
			self.updateLog()
			self.checkCondition()
			overPico = self.getSuggestPico()
			iteration = iteration +1 
			flagString = ""
		
			for green in self.GRs:
				c = color.OKGREEN if self.FLAG[green] else color.FAIL
				done = "Done" if self.FLAG[green] else " BAD"
				flagString  = flagString + green + " " + c+  " " + done +" " + color.ENDC 
			print "Current Flag:\t\t", flagString
			
			### Plot Current Pairs ###
			self.plotLog(show=0,save=1,usemirror = 0)
			self.plotLog(show=0,save=1,usemirror = 1)
			shotString = ""
			for shot in self.SHOT.values():
				shotString = shotString + "%04d" %shot + ","
			shotString2 = "%04d:%04d"%(self.INITSHOT,queue.shotNum()-1)

			queue.runCommand("showlatticealign.py --range " +shotString+" --output " +self.SAVEFOLDER)
			
			if self.reduceFlag():
				queue.runCommand("plot_webcam_report.py --range " +shotString2+" --output " +self.SAVEFOLDER)
				exitinput = raw_input("All Done! Enter r to rest the flags and check all again, or enter anything else to exit:")			
				if exitinput=="r":
					print "Reseting the flags. Will check all shots again."
					self.resetFlag()
				elif (exitinput.isdigit() and len(exitinput) ==6 ):
					print "Overriding the flags. Will check corresponding shots again."
					self.overrideFlag(exitinput)
				else:
					self.plotLog(show=0,save=1,usemirror = 0)
					self.plotLog(show=0,save=1,usemirror = 1)
					break


		

	## Smart start will populate the IR_GOAL and GR from the shots	
	def smartStart(self,shotrange):
		shots = qrange.parse_range(shotrange)
		
		shotsdata = {(1,1,0,0,0,0):"1+2",\
			(1,1,0,1,0,0):"GR1H",\
			(1,1,0,0,1,0):"GR2H",\
			(1,0,1,0,0,0):"1+3",\
			(1,0,1,1,0,0):"GR1V",\
			(1,0,1,0,0,1):"GR3H",\
			(0,1,1,0,0,0):"2+3",\
			(0,1,1,0,1,0):"GR2V",\
			(0,1,1,0,0,1):"GR3V"}
  	
		keys = ["DIMPLE:ir1",\
			"DIMPLE:ir2",\
			"DIMPLE:ir3",\
			"DIMPLE:gr1",\
			"DIMPLE:gr2",\
			"DIMPLE:gr3",\
			"CPP:ax0c",\
			"CPP:ax1c",\
			"CPP:ax0w",\
			"CPP:ax1w"]

		for s in shots:
			report = ConfigObj( 'report'+s+'.INI')  
			temp =[]
			for k in keys:
				temp.append(qrange.evalstr( report, k  ))
			beams = tuple(temp[0:6])
			center = tuple(temp[6:8])
			waist = tuple(temp[8:10])
			pair = shotsdata[beams]
			if ( pair in self.IRs):
				print "Using " + pair + " from shot " + s
				self.IR_GOAL[pair] = [center,waist]
				self.SHOT[pair] = int(s)
			elif ( pair in self.GRs):
				print "Using " + pair + " from shot " + s
				self.GR_POS[pair] = [center,waist]
				self.SHOT[pair] = int(s)
			else: 
				sys.exit("Unexpect IR GR pair condition.")


if __name__ == "__main__":
  	parser = argparse.ArgumentParser(__file__)
  	parser.add_argument('--range', action = "store", default = "",\
        	help="Range of shots to be used for smart start.")
  	parser.add_argument('--analyze', action = "store", default = "analyze # -f -R 100,100,300,300",\
        	help="Command used for analyzing shot. Use # for shot number. Default:analyze # -f -R 100,100,300,300")

  	parser.add_argument('--debug', action = "store_true",\
        	help="Debug mode. Will not actual run shots.")

	args = parser.parse_args()
	queue.DEBUG= args.debug
	auto = autoAlignGr(analyze = args.analyze)
	auto.run(args.range)

