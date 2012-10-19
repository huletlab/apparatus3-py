
import numpy as np
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import matplotlib.dates as mdates


import datetime


def getsiemensdata( datafile, rooms):
  try:
     report = open(datafile,"r")
  except:
     print "ERROR: Can't open Insight Trend Interval Report file"
     print datafile
     exit(1)
  cols=0
  npts=0
  columns=[]
  labels=[]
  for line in report:
     if line.startswith("\"Point"):
        cols=cols+1
        pointname = line.split(',')[1][1:-1]
        print pointname
        if pointname in rooms.keys():
          npts = npts+1
          columns.append(cols-1)
          labels.append(pointname)
  report. close()

  try:
     report = open(datafile,"r")
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
        for i in range(cols):
           first=sub.index("\"") 
           sub=sub[first+1:]
           second=sub.index("\"")
           try:
             datapoint.append( float(sub[:second]))
           except:
             datapoint.append( 0.0 )
           sub=sub[second+1:]
        data.append(datapoint)
        #print "month = %s, day = %s, year = %s, time = %s  %s" % (month,day,year,time,sub)
        #times.append("%02s-%02s-%02s-%s" % (month,day,year,time))
        times.append( datetime.datetime( year, month, day, hour, minute, second) )
     except:
        continue
        

  report.close()
  D = np.array( data )
  print D.shape
  
  return times, D, labels, columns
