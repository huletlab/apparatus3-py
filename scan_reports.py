#!/usr/bin/python

import os
import glob
import time
import argparse

from ConfigParser import SafeConfigParser
import ConfigParser

def eval_test_string( test , inifile):
  tokens = test.split()
  parser = SafeConfigParser()
  parser.read( inifile)
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
      print "%s\t%s\t%s" % (inifile, test, testparse )
#  except:
#    print "Error parsing test string!"
    
  
# EXAMPLE:



if __name__ == "__main__":
  parser = argparse.ArgumentParser('scan_reports.py')
  parser.add_argument('DIR', action="store", type=str, help='path to directory that will be scanned')
  parser.add_argument('TEST', action="store", type=str, help='TEST string to filter out reports, enclose TEST in single quotes')
  
  args = parser.parse_args()


  for rootdir, dirs, files in os.walk( args.DIR ):
    for basename in files:
      if 'report' in basename:
        eval_test_string( args.TEST , os.path.join(rootdir,basename) )

   
    


  


