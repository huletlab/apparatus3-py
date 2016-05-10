#!/usr/bin/env python
#Take an azimuth average data perform inverse abel transformation to get density 

import abel_scikit 
import abel
import numpy as np
import matplotlib.pyplot as plt
import math
import argparse

def inverAbelDat(x,y):
        # First one uses the direct integral
	invx,invy = abel.inverAbel(x,y)
        invx = np.array(invx)
        invy = np.array(invy)

        # Second one uses the radon transform
	x2 = np.append(x[:0:-1],x)
	y2 = np.append(y[:0:-1],y)
	invx2,invy2 = abel_scikit.inverAbel(x2,y2)
        skipn = len(invx2)/2+1
        invx2 = np.array( invx2[skipn:] )
        invy2 = np.array( invy2[skipn:] )

        return (invx, invy),(invx2,invy2)
	
def inverAbel(path,show=1):
	try:
		data = np.loadtxt(path)
	except:
		print "File does not exist"
		return
		
	x =  data[:,0]
	y =  data[:,1]
	invx,invy = abel.inverAbel(x,y)
	x2 = np.append(x[:0:-1],x)
	y2 = np.append(y[:0:-1],y)
	invx2,invy2 = abel_scikit.inverAbel(x2,y2)
	
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax2 = ax.twinx()
	lns1 = ax.plot(x,y,label="Original Azimuthal Data")
	ax.set_xlabel("Pixels")
	ax.set_ylabel("Azimuthal Ave")
	lns2 = ax2.plot(invx,invy,"r--",label="Inv Abel Trans by direct Integral")
	lns3 = ax2.plot(invx2[len(invx2)/2+1:],invy2[len(invx2)/2+1:],"g--",label="Inv Abel Trans by Radon Trans")
	ax2.set_ylabel("Azimuthal Ave Inverse Abel Transfrom")
	lns = lns1+lns2+lns3
	labs = [l.get_label() for l in lns]
	ax.legend(lns, labs, loc=1)
	plt.savefig(path.replace(".AZASCII","_invAbel.png"))
	if (show):
		plt.show()

if __name__ == "__main__":
	parser = argparse.ArgumentParser('inverabel.py')
	parser.add_argument('shot', action="store", type=int, help='The shot number to perform inver abel tranfromation')
	parser.add_argument('--show', action = "store_true", dest='show', help="Show the plot.")

	path= "./%04d"%parser.parse_args().shot + "_datAzimuth.AZASCII"
	inverAbel(path, parser.parse_args().show)
