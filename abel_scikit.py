import matplotlib.pyplot as plt
import numpy
from skimage.io import imread
from skimage import data_dir
from skimage.transform import radon, iradon
from scipy.ndimage import zoom
import math

def inverAbel(x,y):
	x = numpy.array(x)
	y = list(y)
	dx = abs(x[1]-x[0])
	length = len(y)
	#~ print len(numpy.array(y*100))
	sino = numpy.array(y*100).reshape(100,length).transpose()
	reconstruction = iradon(sino)
        #print "len(y) = ", length
        #print "len(reconstruction) = ", len(reconstruction)
        #print "shape(reconstruction) = ", reconstruction.shape
	iry = reconstruction[:,math.ceil(len(reconstruction)*0.5)]/dx
        #print "index = ", math.ceil(len(reconstruction)*0.5) 
        #print "len(iry) = ", len(iry)
        #print "shape(iry) = ", iry.shape
	cropxs = int((length - len(iry))*0.5)-1
	cropxe = cropxs + len(iry)
	irx = x[cropxs:cropxe]
	#print "len(irx), len(iry) =", len(irx),len(iry)
	return  irx,iry


def centerImage(data,cx,cy):
	size = numpy.shape(data)
	 
	lenX = min(cx,size[0]-cx-1)
	lenY = min(cy,size[1]-cy-1)
	print  cx,size[0]-cx-1,cy,size[1]-cy-1
	return data[cx-lenX:cx+lenX+1,cy-lenY:cy+lenY+1]

def aveImage(data1d):
	#average an odd item 1 d array
	length = len(data1d)
	if length%2 ==0:
		return []
	center = length/2
	for i in range(length/2):
		xave = (data1d[center-i-1]+data1d[center+i-1])*0.5
		data1d[center-i-1] =xave 
		data1d[center+i-1]=xave
	return data1d
