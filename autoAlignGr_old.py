#!/usr/bin/python
from queue import *
#DEBUG.append(0)
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'


GRs = ["GR1H","GR1V","GR2H","GR2V","GR3H","GR3V"]
IRs = ["1+2","2+3","1+3"]
keysToRead = ("CPP:ax0w","CPP:ax1w","CPP:ax0c","CPP:ax1c")
iniPath = os.path.abspath(__file__).replace(".py",".ini")
DATA = ConfigObj(iniPath)
POS_CONDITION = 1.0
MAX_ITERATION = 10.0

FLAG = {}
GR_SLOPE={}
Goal = {}
GR_GOAL={}
GR={}
SHOT={}
ANALYSIS = "analyze # -f -R 100,100,300,300"
for green in GRs:
	FLAG[green] = 0
	GR_SLOPE[green]=tuple([ float(i) for i in DATA["SLOPE"][green]])
	SHOT[green]=-1

for ir in IRs:
	SHOT[ir]=-1

def reduceFlag(flag):
	temp =1
	for f in flag.values():
		temp = temp *f
	return temp 

def parseOverride(overList):
	override = ""
	overrideKey = ["DIMPLE:ir1","DIMPLE:ir2","DIMPLE:ir3","DIMPLE:gr1","DIMPLE:gr2","DIMPLE:gr3" , "DIMPLELATTICE:knob01"]
	for key,value in zip(overrideKey,overList):
		override = override + key + "\t%f\t"%(float(value)) 
	override = override +"DIMPLELATTICE:imgdet\t-110\n"
	return override

def getSuggestPico(gr,goal,slope,flag):
	global GRs 
	suggestPico={}
	for i,green in enumerate(GRs):
		pos = gr[green][0]
		goalpos = goal[green][0]
		delta = (pos[0]-goalpos[0],pos[1]-goalpos[1])
		sl = slope[green]
		for i,s in enumerate(sl):
			if s != 0:
				suggestPico[green] = int(delta[i]*s) if (not flag[green]) else 0
	return suggestPico

def parseSuggestPico(sp):
	override = "PICO:move\t1\t"
	for key,pico in sp.iteritems():
		if pico != 0 :
			override = override + "PICO:m" + key +"\t%d\t"%pico
	return override

def checkAllGr(gr,goal,flag,data,pico="",analyze = ANALYSIS,force=1):
	gr_to_check = []
	keys_to_read = ("CPP:ax0w","CPP:ax1w","CPP:ax0c","CPP:ax1c")
	global GRs 
	if (not force):
		for green  in GRs:
			if (not flag[green]):
				gr_to_check.append(green) 				
	else:
		gr_to_check = GRs
	for i,green in enumerate(gr_to_check):
		print "Checking " + green
		movepico = pico if (i ==0 ) else ""
		override = movepico + parseOverride(data["OVERRIDE_GR"][green])
		shotnum,result = runShotWaitAnalyzeRead(override,analyzeCommand=analyze, keys=keys_to_read)
		gr[green] = [(float(result["CPP:ax0c"]),float(result["CPP:ax1c"])),(float(result["CPP:ax0w"]),float(result["CPP:ax1w"]))]
		SHOT[green] = shotnum
	
def checkCondition(gr,goal,flag):	
	global GRs 
	global POS_CONDITION
	for green in GRs:
		g_pos = gr[green][0]
		i_pos = goal[green][0]
		d_pos = [g_pos[0]-i_pos[0],g_pos[1]-i_pos[1]]
		flag[green] = 0 if ((abs(d_pos[0]) > POS_CONDITION) or (abs(d_pos[1])> POS_CONDITION)) else 1
				
##	Check Pair and Find Goals	##
### Should make this part as a function later #####

for ir in IRs:
	print "Checking " + ir 
	override = parseOverride(DATA["OVERRIDE_IR"][ir])
	shotnum,result = runShotWaitAnalyzeRead(override,analyzeCommand=ANALYSIS, keys=keysToRead)
	Goal[ir] = [(float(result["CPP:ax0c"]),float(result["CPP:ax1c"])),(float(result["CPP:ax0w"]),float(result["CPP:ax1w"]))]
	SHOT[ir] = shotnum

wc = 1.0
warning = 0
warningstring = ""

for key,pair in Goal.iteritems():
	for akey,apair in Goal.iteritems():
        	if key != akey:
                	d0 = abs(Goal[key][0][0]-Goal[akey][0][0])
                	d1 = abs(Goal[key][0][1]-Goal[akey][0][1])
			if d0>wc:
                        	warning = 1
                        	warningstring = warningstring + key +" and " +akey+" separate in 0 by %.2f pixel.\n"%d0
                	if d1>wc:
                        	warning = 1
                        	warningstring = warningstring + key +" and " +akey+" separate in 1 by %.2f pixel.\n"%d1


if warning:
	print warningstring
	print "##########	IR pairs are not good! .		##########"
else:
	print "##########	IR pairs are good continue with green.	##########"

exitinput = raw_input("Press Enter to continue .. or enter anything to exit")

if exitinput:
	exit(1)

for green in GRs:
	GR_GOAL[green]=Goal[DATA["GOAL"][green]]

##	Cheking GR Pair		##
### Should make this part as a function later #####

overPico=""
iteration = 0
suggestPico = {}
color = bcolors()

while ((not reduceFlag(FLAG)) & (iteration<MAX_ITERATION)):
	print color.OKBLUE,"\nCheck Green -- Iteration %d" %(iteration+1) ,color.ENDC
	
	if suggestPico:
		picoString =""
		for green in GRs:
			c = color.OKGREEN if (suggestPico[green]==0) else color.FAIL
			picoString  = picoString + green + " " + c+  " " + "%4d"%suggestPico[green] +" " + color.ENDC 
		print "Move Pico:\t\t",picoString
	
	exitinput = raw_input("Press Enter to continue .. or enter anything to exit")
	if exitinput:
		exit(1)

	checkAllGr(GR,GR_GOAL,FLAG,DATA,overPico,analyze = ANALYSIS,force=(not iteration))
	checkCondition(GR,GR_GOAL,FLAG)
	suggestPico = getSuggestPico(GR,GR_GOAL,GR_SLOPE,FLAG)
	overPico = parseSuggestPico(suggestPico)
	iteration = iteration +1 
	flagString = ""

	for green in GRs:
		c = color.OKGREEN if FLAG[green] else color.FAIL
		done = "Done" if FLAG[green] else " BAD"
		flagString  = flagString + green + " " + c+  " " + done +" " + color.ENDC 
	print "Current Flag:\t\t", flagString
	
	### Plot Current Pairs ###
	shotString = ""
	for shot in SHOT.values():
		shotString = shotString + "%d" %shot + ","
	runCommand("showlatticealign.py --range " +shotString)


