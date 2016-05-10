import math 
import matplotlib.pyplot as plt 
import numpy as np
from hansenlaw import hansenlaw_transform

def Abel(dataX,dataY):
	Y = []
	dx = dataX[1]-dataX[0]
	for i,r in enumerate(dataX):
		
		transY = 0
		tempY = dataY[i]
		for x,y in zip(dataX[i:],dataY[i:]) :
			x= x +dx/2 # shift half division to avoid singularity
			transY = transY +2*y*x/(x**2-r**2)**0.5*dx
		Y.append(transY)
	#~ Y[0] = Y[1]
	
	return dataX, Y


def inverAbel(dataX,dataY, pyabel = True):
	if pyabel:
		dr = dataX[1]-dataX[0]
		Y = hansenlaw_transform(dataY, dr)
	else:
	"""
	The old code
	"""
		Y = []
		dx = dataX[1]-dataX[0]
		dataXoffset = np.array(dataX)-dataX[0]
		for i,r in enumerate(dataXoffset):
			
			inverY = 0
			tempY = dataY[i]
			for x,y in zip(dataX[i:],dataY[i:]) :
				x= x +dx/1e10 # shift half division to avoid singularity
				dy = y -tempY
				inverY = inverY - 1/math.pi*dy/(x**2-r**2)**0.5
				tempY = y
			Y.append(inverY)
	return dataX, Y

    
