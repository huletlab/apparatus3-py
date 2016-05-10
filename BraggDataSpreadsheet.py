#!/usr/bin/python

import os 


# Get the spreadsheet 
bds_file  = '/lab/data/app3/2013/BraggDataSpreadsheet-Sheet1.tsv'

verbose=False

with open(bds_file,'r') as bds:
    i = 0 
    for line in bds:
        row = line.split('\t')
        if row[2].startswith('B') and int(row[1]) > 130614:
            i = i + 1
            if i > 200: 
              break
            try:
              setnum  = row[0]
              date    = row[1] 
              basekey = row[5]
              shots   = row[4]
              save    = row[2] 
  
              os.chdir('/lab/data/app3/2013/' + date[0:4] + '/' + date)
              savepath = 'plots2/'
              if not os.path.exists(savepath):
                os.makedirs(savepath)
              if verbose:
                print  
                print 'Changed directory to: ', os.getcwd()
                print 'Working on set #', save 


              # SHOWBRAGG
              #command = 'showbragg2.py ' + basekey + \
              #          ' ANDOR:exp HHHEIGEN:andor2norm  ANDOR1EIGEN:signal ANDOR2EIGEN:signal --output '\
              #          + savepath + save + '.png --range ' + shots + ' --concise'
              #if verbose:
              #  print command
              #os.system( command ) 

              # QRANGE
              #print  '\n',row[2]
              #command = 'qrange.py ' + shots + ' DIMPLELATTICE:knob1 DIMPLE:image DIMPLE:ir1pow0 DIMPLE:allirpow' 
              #os.system( command )

              # GETDATE
              command = r"stat --printf='%y\n' report" + shots[0:4] + '.INI'
              os.system(command)
              

            except Exception as e:
              print e
         
