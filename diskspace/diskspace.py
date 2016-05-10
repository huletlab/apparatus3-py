#!/usr/bin/python
import subprocess
import re
from gmail import *
def checkDisk(threshold=80,ignore=["/media/BackupData","/media/dataBackupEMTsOLD"]):
	proc = subprocess.Popen("df -h", shell=True,stdout=subprocess.PIPE)
	proc.wait()
	output =  proc.stdout.read()
	lines =  output.split('\n')
	wString = ""
	for line in lines[1:]:
		data = line.split()
		if len(data) >0:
			print data
			size = data[4].split("%")[0]
			print size
			if (int(size)>threshold)&(data[5] not in ignore): 
				wString = wString + "Disk usage of " +data[5] + " is " + size + "%\n"
				print data[5]
	return wString,output	

if __name__ == "__main__":
	os.getcwd()
        inifile = os.path.join(os.path.dirname(__file__),'diskspace.ini')
        print inifile
        config = ConfigObj(inifile)

	usr = config["Gmail"]["usr"]#"apparatus3huletlab"
	pwd = config["Gmail"]["pwd"]#"cesium137"
	rec = [pair[1] for pair in config["WarningList"].items()]
	thr = int(config["WarningThreshold"]["threshold"])
	wString,output = checkDisk(thr)
	if wString:
		sendGmail(usr,pwd,rec,"Disk Space Warning",wString+"<br><br>Here is the df -h output:<br><br>"+output.replace("\n","<br>"))
	#else:
	#	sendGmail(usr,pwd,rec,"Disk Space Warning","No Problem")
