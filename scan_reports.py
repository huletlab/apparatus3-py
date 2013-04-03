#!/usr/bin/python

import os
import glob
import time
import argparse

import sys


from configobj import ConfigObj

import qrange
import datetime
absstart = datetime.datetime( 1903, 12, 31, 18, 00, 00 )

def sortedWalk(top, topdown=True, onerror=None):
   from os.path import join,isdir,islink

   names = os.listdir(top)
   names.sort()
   dirs, nondirs = [], []
 
   for name in names:
      if isdir(os.path.join(top,name)):
         dirs.append(name)
      else:
         nondirs.append(name)

   if topdown:
      yield top, dirs, nondirs
   for name in dirs:
      path = join(top,name)
      if not os.path.islink(path):
         for x in sortedWalk( path, topdown, onerror):
            yield x
   if not topdown:
      yield top, dirs, nondirs

def eval_test_string( test , inifile, keys):
  try:
    tokens = test.split()
    parser = SafeConfigParser()
    parser.read( inifile)
  except:
    return
  valid = True
  for i,tok in enumerate(tokens):
    if ':' in tok:
      try:
        sec = tok.split(':')[0]
        key = tok.split(':')[1]
        tokens[i] = parser.get( sec, key )
      except:
        valid = False
  if valid:
    testparse = " ".join(tokens)
    if eval( testparse ):
      if keys == None:
         print "%s\t%s\t%s" % (inifile, test, testparse )
      else:
         err=False
         line=''
         line_=''
         errmsg=''
         for pair in keys: 
            sec = pair.split(':')[0]
            key = pair.split(':')[1]
            try:
              if key == 'abstime':
                 val = parser.get( sec, key) 
                 lstr = '%s\t' % val
              else: 
                 fval = float( parser.get( sec, key ))
                 
                 if fval > 1e5 or fval < -1e5:
                   lstr = '%.3e\t' % fval
                 else: 
                   lstr = '%6f\t' % fval
            except KeyError:
              err= True
              errstr = '#...Failed to get %s:%s from #%s\n' % (sec, key, shot)
              errmsg = errmsg + errstr
              lstr = 'nan\t\t' 
            line_ = line_ + lstr
              
         if not err:
            print line_
         else: 
            print errstr 
            print line_
         
#  except:
#    print "Error parsing test string!"
    
  
# EXAMPLE:



if __name__ == "__main__":
  parser = argparse.ArgumentParser('scan_reports.py')
  parser.add_argument('DIR', action="store", type=str, help='path to directory that will be scanned')
  parser.add_argument('TEST', action="store", type=str, help='TEST string to filter out reports, enclose TEST in single quotes')
  parser.add_argument('--keys', nargs='*', help='keys to be shown at the output')
  
  args = parser.parse_args()

  if args.keys != None:
    rawdat='%s\n' % ('\t'.join(args.keys))
    rawdat='#Date\t\t' + rawdat
    print rawdat,
  for rootdir, dirs, files in sortedWalk( args.DIR ):
    for basename in files:
      if 'report' in basename:
        reportpath = os.path.join(rootdir, basename)
        try:
          report = ConfigObj(reportpath)
        except:
          print "Could not load report %s" % reportpath
          continue
        keys = args.keys
        if qrange.evalstr( reportpath, args.TEST):
            if keys == None:
               print "%s\t%s\t is True" % (reportpath, args.TEST )
            else:
               err=False
               line=''
               line_=''
               errmsg=''

               try:
                 abstime = float( report['SEQ']['abstime'])
                 date =  absstart + datetime.timedelta( seconds = abstime)
                 line_ = date.strftime('%a %d-%b-%Y %H:%M\t\t')
               except KeyError:
                 err= True
                 errstr = '#...Failed to get SEQ:abstime from #%s\n' % (reportpath)
                 sys.stderr.write(errstr)
                 continue

               for pair in keys: 
                  
                  sec = pair.split(':')[0]
                  key = pair.split(':')[1]
                  try:
                    if key == 'abstime':
                       val = report[sec][key]
                       lstr = '%s\t' % val
                    else: 
                       fval = float( report[sec][key])
                       if fval > 1e5 or fval < -1e5:
                         lstr = '%.3e\t' % fval
                       else: 
                         lstr = '%6f\t' % fval
                  except KeyError:
                    err= True
                    errstr = '#...Failed to get %s:%s from #%s\n' % (sec, key, reportpath)
                    errmsg = errmsg + errstr
                    lstr = 'nan\t\t' 
                  line_ = line_ + lstr
                    
               if not err:
                  print line_
               else: 
                  sys.stderr.write( errstr )
                  sys.stderr.write( line_ )
         
        
         
      #  eval_test_string( args.TEST , os.path.join(rootdir,basename) , args.keys)

   
    


  


