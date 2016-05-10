#!/usr/bin/python

import os
import glob
import time
import re
import pprint

def shot_from_path( path ):
    try:
        name = os.path.basename(path)
        #Find a string of four digits in name 
        shot = re.findall( '\d\d\d\d', name)
        if len(shot) > 1:
            raise ValueError
        else:
           shot = shot[0] 
        return int(shot)
    except:
        print "Error obtaining shot number from path: %s" % path
        return None


def get_all_shots( path, exprs):
    sets = [ [ shot_from_path(p) for p in  glob.glob( path + e ) ]  for e in exprs]

    # Print this for debugging
    #for i,s in enumerate(sets): 
    #   s.sort()
    #   print 
    #   print exprs[i]
    #   print s

    # Get union of shots from all exprs
    s0 = set(sets[0])
    for s in sets:
      s0 = s0 | set(s)
    s0 = list(s0)
    s0.sort()
    return s0

def get_last_shot( path, exprs ):
    allshots = get_all_shots( path, exprs)
    if allshots[-1] == 9999:
	last = 9999
	for shot in allshots:
		if shot >8000:
			return last
		last = shot	

    return allshots[-1]
   

    

def waitforit( path, expr ):
    current = glob.glob(path + expr) 
    #print "waiting..."
    #for i in range(20):
    while ( True ) :
      try:
        time.sleep(2)
        new = glob.glob( path + expr ) 
        diff = []
        for f in new: 
           if not f in current:
             diff.append(f)
        if len(diff) > 0:
           return diff
      except KeyboardInterrupt:
        exit(1)

def waitforit_list( path, exprs, sleeptime = 2, current = None):
    if current is None:
        current = get_all_shots( path, exprs) 
    #print "waiting..."
    #for i in range(20):
    while ( True ) :
      try:
        time.sleep(sleeptime)
        new = get_all_shots( path, exprs)
       
        #print new 
        #print current 

        diff = []
        for f in new: 
           if not f in current:
             diff.append(f)
        if len(diff) > 0:
           return diff
      except KeyboardInterrupt:
        exit(1)



if __name__ == "__main__":
  #diff = waitforit( '/lab/data/app3/2012/1206/120621/', '????atoms.fits' )
  #print diff
  new = waitforit_list( os.getcwd(), ['/????atoms.manta', '/????atoms.fits', '????atoms_andor2.fits'], sleeptime = 3 )
  print
  print "List of shots to analyze:", new
    


  


