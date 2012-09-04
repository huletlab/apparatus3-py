import numpy
from scipy import stats

def statdat( dataset , Xcol, Ycol):
  """Set is a numpy array of data.  This fuction looks for rows
     that have Xcol in common and for each different value of Xcol
     gives out the mean and standard error of the maean.  
     
     The result is returned as a numpy array.   
  """
  out =[]
  while dataset.shape[0] != 0: 
    #print "# of rows in array = %d" % set.shape[0]
    #print set
    Xval = dataset [0, Xcol]
    Yval = []
    to_delete = [] 
    for i in range( dataset.shape[0] ) :
      row = dataset[i,:]
      if row[Xcol] == Xval:
        to_delete.append(i)
        Yval.append( row[Ycol]) 
    dataset = numpy.delete( dataset, to_delete , 0)
    #print "# of rows in array = %d" % set.shape[0]
    #print set 
    Yarray = numpy.array( Yval)
    mean = numpy.mean(Yarray)
    stddev = numpy.std(Yarray)
    serror = stats.sem(Yarray)
    #print Yval
    out.append( [Xval, mean, serror] ) 
  return numpy.array( out )


