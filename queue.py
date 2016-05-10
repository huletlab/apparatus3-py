__author__= "Ernie Yang"
DEBUG = []
import os,sys
import subprocess as subp
from configobj import ConfigObj
import time
import random
import time

def correctWinPath(path):
	if os.name == 'posix':
		return path.replace('\\','/').replace('L:','/lab')
	else:
		return path

settingPath = correctWinPath("L:\\software\\apparatus3\\settings.INI")
setting = ConfigObj(settingPath)
queuePath =  correctWinPath(setting['COMMS']['paramqueue'])
shotNumPath = correctWinPath(setting['COMMS']['runnum'])
saveDirPath = correctWinPath(setting['COMMS']['savedir'])

def appendLineToQueueOld(string,queueFile=queuePath):
	global DEBUG
	if (DEBUG):
		queueFile =  queueFile.replace(".dat","_test.dat")
	with open (queueFile, "a+") as queue:
		queuestring = queue.read()
		if not queuestring.endswith('\n'):
			queue.write('\n')
		queue.write(string)

def appendLineToQueue(string,queueFile=queuePath):
	global DEBUG
	if (DEBUG):
		queueFile =  queueFile.replace(".dat","_test.dat")
	maxT = 10
	t = 0
	while t<(maxT):
			t = t +1
			try:
				queue = open (queueFile, "a+") 
				queuestring = queue.read()
				if not queuestring.endswith('\n'):
					queue.write('\n')
				queue.write(string)
				break
			except:
				print "Fail to write to queue file. Atempting(%d/%d)"%(t,maxT)
				time.sleep(1)
	


def clearQueue(queueFile=queuePath):
	global DEBUG
	if (DEBUG):
		queueFile =  queueFile.replace(".dat","_test.dat")
	open(queueFile,"w").close()


def shotNum(shotNumFile=shotNumPath):
	with open (shotNumFile, "r") as shotNum:
		return int(shotNum.read())
		
def runCommand(command):
	process = subp.Popen(command,shell=True,stdout=subp.PIPE,stderr=subp.PIPE)
	output,error = process.communicate()
	return output,error
#def parseQueueLine(dic)

def readReport( keys, shotnum = -1, saveDir = saveDirPath ):
	global DEBUG
	if ( not DEBUG):
		if shotnum == -1:
			shotnum = shotNum() - 1	
			with open (saveDir, "r") as sd:
				saveDir = correctWinPath(sd.read())
		else:
			saveDir = os.getcwd()+"/" #
		#print "Reading Report" +saveDir+'report%04d.INI'%(shotnum)
		report = ConfigObj(saveDir+'report%04d.INI'%(shotnum))
		values = {}
		for key in keys:
			sec,k = key.split(':') 
			values[key] = report[sec][k]
	else:
		values = {}
		for key in keys:
			sec,k = key.split(':') 
			values[key] =  random.random() *100


	return values 

def waitUntilShotIsDone(shot,timeInterval=1):
	print "Waiting for shot %04d"%(shot)
	while (shot >= shotNum()):
		time.sleep(timeInterval)
	print "Shot %04d is done"%(shot)

def runShotWait(override="QUEUE:dummy\t1\n"):
	global DEBUG
	if ( not DEBUG) :
		clearQueue()
		shotnum = shotNum()
		appendLineToQueue(override)
		waitUntilShotIsDone(shotnum)
	else:
		clearQueue()
		appendLineToQueue(override)
		shotnum = -1

	return shotnum

def runShotWaitAnalyzeRead(override="QUEUE:dummy\t1\n",analyzeCommand="analyze # -f", keys=('SEQ:shot')):
	global DEBUG
	shotnum = runShotWait(override)
	command = analyzeCommand.replace("#","%04d"%(shotnum))
	print command 
	if (not DEBUG):
		output,error = runCommand(command)
		print output
	else:
		#output,error = runCommand("cp ../131220/report1864.INI ./report%04d.INI"%(shotnum))
		output,error = runCommand('ls')
		#print output
	
	return  shotnum,readReport(keys,shotnum) 
	
if __name__ == "__main__":
	#clearQueue()
	#print shotNum()
	#print runCommand('cd `gotodat`;pwd')
	runShotWaitAnalyzeRead()
