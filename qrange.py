#!/lab/software/epd-7.0-1-rh5-x86/bin/python

import sys
from configobj import ConfigObj
import numpy

from StringIO import StringIO

def parse_rangeWIN(rangestr):
  shots=[]
  l = rangestr.split(',')
  for token in l:
    if token.find(':') > -1: 
      sh0 = int(token.split(':')[0])
      shf = int(token.split(':')[1])
      if shf < sh0:
        sys.stderr.write("\n --->  RANGE ERROR: end of range is smaller than start of range\n\n")
        return
      for num in range(sh0,shf+1):
        numstr = "%04d" %num
        shots.append(numstr)
    elif token.find('-') == 0:
      l2 = token.split('-')[1:]
      for shot in l2:
        numstr = "%04d" % int(shot)
        if numstr in shots:
          shots.remove(shot)

  return shots

def qrange(dir,range,keys):
  fakefile=""
  shots=parse_rangeWIN(range)
  errmsg=''
  rawdat='#%s%s\n' % ('SEC:shot\t',keys.replace(' ','\t'))

  for shot in shots:
    report = dir + 'report' + shot + '.INI'
    report = ConfigObj(report)
    if report == {}:
      errmsg=errmsg + "...Report #%s does not exist in %s!\n" % (shot,dir)
      continue
    fakefile = fakefile + '\n'
    rawdat = rawdat + '\n%s\t\t' % shot
    line=''
    line_=''
    err=False
    for pair in keys.split(' '):
      sec = pair.split(':')[0]
      key = pair.split(':')[1]
      try:
        val = report[sec][key]
        line = line + val + '\t'
        fval = float(val)
        if fval > 1e5 or fval < -1e5:
          lstr = '%.3e\t\t' % fval
        else: 
          lstr = '%.4f\t\t' % fval
        line_ = line_ + lstr
      except KeyError:
        err= True
        errstr = '...Failed to get %s:%s from #%s\n' % (sec, key, shot)
        errmsg = errmsg + errstr
    if not err:
      fakefile = fakefile + line
      rawdat = rawdat + line_
   
  a = numpy.loadtxt(StringIO(fakefile))
  print errmsg
  return a, errmsg, rawdat
 
 

# --------------------- MAIN CODE  --------------------#


if __name__ == "__main__":
  linux = True
  Windows = False
  if linux:
    prefix = "/lab/"
  elif Windows:
    prefix = "L:/"
  else:
    print " ---> Unrecognized operating system!!"  
    exit(1)
 
  a, errmsg, rawdat = qrange(prefix + 'data/app3/2011/1107/110725/','8888:8890','SEQ:shot CPP:nfit')
  print rawdat
