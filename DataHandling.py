import numpy as np
from uncertainties import ufloat,unumpy

from scipy import stats
import os, sys
sys.path.append('/lab/software/apparatus3/py')
import statdat, qrange

from scipy.interpolate import interp1d


def data_fetch( datakeys, gdat, **kwargs ):
    save = kwargs.get('save', False) 
    fmt = kwargs.get('fmt', '%9.4g') 
    print "Fetching data..."
    for k in gdat.keys():
        try:
            gdat[k]['data'] = np.loadtxt(k+'.dat')
            print "Loaded %s succesfully." % k 
        except:
            data, errmsg, rawdat = qrange.qrange_eval( gdat[k]['dir'], gdat[k]['shots'], datakeys) 
            if save:
                np.savetxt(k+'.dat', data, fmt=fmt)
            gdat[k]['data'] = data
    print "Done."
  
    K = lambda key: datakeys.index(key)
    return gdat, K 


def data_pick( dat, conds, K):
    dat1 = np.array(np.copy(dat))
    #return dat1	
    for c in conds: 
        Kc = K(c[0]) 
        cval = c[1]
        if len(dat1) > 0 :
            try:
                if len(c) == 2:
                    dat1 = dat1[ dat1[:,Kc] == cval ]  
                elif len(c) == 3:
                    if c[2] == '>': 
                        dat1 = dat1[ dat1[:,Kc] > cval ] 
                    elif c[2] == '<': 
                        dat1 = dat1[ dat1[:,Kc] < cval ] 
                    elif c[2] == '=': 
                        dat1 = dat1[ dat1[:,Kc] ==  cval ] 
                    elif c[2] == '>=': 
                        dat1 = dat1[ dat1[:,Kc] >=  cval ] 
                    elif c[2] == '<=': 
                        dat1 = dat1[ dat1[:,Kc] <=  cval ] 
                    elif c[2] == '!=': 
                        dat1 = dat1[ dat1[:,Kc] != cval ] 
            except:
                print "Condition failed:"
                print "key index = ", Kc
                print "looking for cval = ",cval 
                print "in ", dat1
                raise
    return dat1 

def data_ratio( dat, cond1, cond2, K, Xkey, Ykey, **kwargs): 
    # First get the data for each condition
    dat1 = data_pick( dat, cond1, K) 
    dat2 = data_pick( dat, cond2, K) 

    # Then determine the x values they have in common 
    x1= np.unique(dat1[:,K(Xkey)])
    x2= np.unique(dat2[:,K(Xkey)])
    set1 = set( x1.tolist())
    set2 = set( x2.tolist()) 
    common = sorted(list( set1 & set2 ))

    # Decide which operation will be performed on the two sets
    operation = kwargs.get( 'operation', 'ratio') 
  
    # Then loop through the common x values to get the 
    # quantities of interest
    num = dat1 ; den = dat2 ; 
    rval = []; rerr = [] ; nsamples=[]
    for c in common:
        numi = num[ num[:,K(Xkey)] == c ][ :, K(Ykey) ] 
        deni = den[ den[:,K(Xkey)] == c ][ :, K(Ykey) ]
        ns = max( len(numi), len(deni)) 
        if operation == 'ratio':
            val =  ufloat(( np.mean(numi), stats.sem(numi) )) / ufloat(( np.mean(deni), stats.sem(deni) ))
        elif operation == 'sum': 
            val =  ufloat(( np.mean(numi), stats.sem(numi) )) + ufloat(( np.mean(deni), stats.sem(deni) ))
        else: 
            raise Exception('undefined operation in DataHandling.data_ratio' ) 
            
        rval.append( val.nominal_value )
        rerr.append( val.std_dev() )
        # nsamples.append( ns )  # Changed the behaviour of nsamples 140428
        nsamples.append( [len(numi), len(deni)] ) 

    rval = np.array( rval )  
    rerr = np.array( rerr )
    xc = np.array(common)
    return xc, rval, rerr, nsamples


def average_errdata( sets ):
    xsets = []
    for s in sets:
        xsets.append( set( s[0].tolist()) ) 
      
    # Determine the x values they are common for all sets
    common = set.intersection( *xsets )
  
    def inset(x, xset):
   
        if x in xset:
            return True
    
    X=[]; Y=[]; YERR=[]; 
    for s in sets:
        xs = []; ys = []; yerrs=[]; 
        for i in range( len(s[0] )): 
            if s[0][i] in common:
                xs.append( s[0][i] ) 
                ys.append( s[1][i] ) 
                yerrs.append( s[2][i] )

        xs = np.array( xs ) 
        ys = np.array( ys )
        yerrs = np.array( yerrs) 
 
        index = np.argsort (xs ) 
        X.append( xs[ index ] )
        Y.append( ys[ index ] )
        YERR.append( yerrs[ index] )

    X = np.array(X) 
    Y = np.array(Y) 
    YERR = np.array(YERR)  

    #print "\nX=" 
    #print X 
    #print "\nY="
    #print Y 
    #print "\nYERR=" 
    #print YERR
 
    # Perform weighted average  
    weights = 1/ np.power( YERR, 2. )
    y_wav = np.sum( Y * weights , axis=0) / np.sum(weights, axis=0 ) 
    y_err_wav = 1./ np.sqrt( np.sum(weights, axis=0) )  
     
    #print "\nWeighted average" 
    #print y_wav
    #print "\nError"
    #print y_err_wav 
 
    return X[0], y_wav, y_err_wav


# PLOTTING FUNCTIONS 
 
def plotkey( ax, gdict, K, fx, xkey, ykey, dat, base, **kwargs):
    try:
        defmarker = gdict['marker']
    except:
        defmarker = 'o'  
    marker = kwargs.get( 'marker', defmarker)
    ms     = kwargs.get( 'ms', 5. ) 
    mew    = kwargs.get( 'mew', 1.0 ) 
    mec    = kwargs.get( 'mec', gdict['ec'])
    mfc    = kwargs.get( 'mfc', gdict['fc'])
    ew     = kwargs.get( 'ew', 1.0 )
    ecap   = kwargs.get( 'ecap', 0. )
    save   = kwargs.get( 'save', False )

    exceptions = kwargs.get( 'exceptions', False) 

    raw        = kwargs.get( 'raw', True )
    raw_offset = kwargs.get( 'raw_offset', 0.) 

    labelstr = kwargs.get( 'labelstr', gdict['label'] ) 
  
    xkey0 = xkey
    ykey0 = ykey 
    xkey = K(xkey)
    ykey = K(ykey)   

    if kwargs.get('use_stddev', False):
        error_index = 2 # standard deviation
    else:
        error_index = 3 # standard error


    discard = kwargs.pop('discard',None)
    if discard is not None:
        if 'y>' in discard.keys():
            dat = dat[ dat[:,ykey] < discard['y>'] ] 
        if 'y<' in discard.keys():
            dat = dat[ dat[:,ykey] > discard['y<'] ] 
   
           
    
 

    try:
        both_offset = kwargs.get('both_offset', 0.)

        yf = kwargs.get( 'yf', None)
        if yf is not None: 
            ydict = kwargs.get('yf_kwargs', {} )  
            yf_usex = kwargs.get('yf_usex', False)
            if yf_usex is True:
                ydict['x'] = fx(xc)


        if raw:
            rawcolor = kwargs.pop('rawcolor', 'gray') 
            rawalpha = kwargs.pop('rawalpha', 0.5) 
            rawms = kwargs.pop('rawalpha', 4.5)

            xraw = fx(dat[:, xkey] ) + raw_offset + both_offset
            yraw = dat[:,ykey]/base
            if yf is not None: 
                yraw = yf( yraw,  **ydict )
                
            ax.plot( xraw, yraw , '.',
                     marker='.', ms=rawms,\
                     color=rawcolor, alpha=rawalpha) 

        datS = statdat.statdat( dat, xkey, ykey, **kwargs)

        xplot = fx(datS[:,0])
        yplot = datS[:,1]/base
        yploterr = datS[:,error_index]/base
        if yf is not None: 
            yunc = unumpy.uarray(( yplot, yploterr ))
            yunc = yf( yunc, **ydict ) 
            yplot = unumpy.nominal_values( yunc  ) 
            yploterr = unumpy.std_devs( yunc )
         
        ax.errorbar( xplot+both_offset,  yplot, yerr=yploterr,\
                      capsize=ecap, elinewidth=ew,\
                      fmt='.', ecolor=mec, mec=mec,\
                      mew=mew, marker=marker, mfc=mfc, ms=ms,\
                      label=labelstr)

        guide = kwargs.get( 'guide', False)
        if guide:
            guide_color = kwargs.get('guide_color', mec)
            f = interp1d( xplot, yplot, kind='linear')
            xnew = np.linspace( xplot.min(), xplot.max(), 120)
            ax.plot( xnew, f(xnew), color=guide_color, zorder = 1.1 )

 
        if save:
            fname = xkey0.replace(':','_') + '_' +  labelstr[-3:] + '.rawdat' 
            X =  np.transpose(np.vstack( ( xplot, yplot, yploterr )))
            np.savetxt( fname, X, fmt='%10.2f', delimiter='\t', newline='\n')

        if kwargs.get('return_raw', False):
            return xraw, yraw

        return  xplot, unumpy.uarray(( yplot, yploterr ))

    except:
        print "Exception occured in plotkey %s vs %s"% (ykey0,xkey0) 
        if exceptions:
            raise

def plotkey_relerr( ax, gdict, K, fx, xkey, ykey, dat, base, **kwargs):
    try:
        defmarker = gdict['marker']
    except:
        defmarker = 'o'  
    marker = kwargs.get( 'marker', defmarker)
    ms     = kwargs.get( 'ms', 5. ) 
    mew    = kwargs.get( 'mew', 1.0 ) 
    mec    = kwargs.get( 'mec', gdict['ec'])
    mfc    = kwargs.get( 'mfc', gdict['fc'])
    save   = kwargs.get( 'save', False )

    exceptions = kwargs.get( 'exceptions', False) 

    raw        = kwargs.get( 'raw', True )
    raw_offset = kwargs.get( 'raw_offset', 0.) 

    labelstr = kwargs.get( 'labelstr', gdict['label'] ) 
  
    xkey0 = xkey
    ykey0 = ykey 
    xkey = K(xkey)
    ykey = K(ykey)    

    try:
        datS = statdat.statdat( dat, xkey, ykey )

        Ystderr = datS[:,3]/base
        Ystddev = datS[:,2]/base
        X = datS[:,0]
        Y = datS[:,1]/base
    
        def meanY( x ):
            try:
                index = np.where( X == x )[0][0]
                assert isinstance( index, int ) 
                return Y[index] 
            except:
                print "Error finding mean value of Y at X = ", x 
                raise 
        if raw:
            datY = dat[:,ykey]/base 
            datX = dat[:,xkey] 
            datYnormed = np.array([ datY[i] / meanY( datX[i] ) \
                                    for i in range(len(datX)) ] )

            rawcolor = kwargs.pop('rawcolor', 'gray') 
            ax.plot( fx(datX)+raw_offset, datYnormed, '.',\
                     marker='.', ms=4.5,\
                     color=rawcolor, alpha=0.5) 

        ax.plot( fx(X), 100.*2.*Ystderr/Y, '.',\
                 mec=mec, mew=mew, marker=marker, mfc=mfc, ms=ms,\
                 label=labelstr)
        #ax.plot( fx(X), 100.*Ystddev/Y, '.',\
        #         mec=mec, mew=mew, marker=marker, mfc=mfc, ms=ms,\
        #         label=labelstr)

        if save:
            fname = xkey0.replace(':','_') + '_' +  labelstr[-3:] + 'RELERR.rawdat' 
            X =  np.transpose(np.vstack( ( fx(datS[:,0]), datS[:,1]/base, datS[:,3]/base )))
            np.savetxt( fname, X, fmt='%10.2f', delimiter='\t', newline='\n')
        return  fx(datS[:,0]), unumpy.uarray(( datS[:,1]/base, datS[:,3]/base )) 
    except:
        print "Exception occured in plotkey %s vs %s"% (ykey0,xkey0) 
        if exceptions:
            raise
        return None


def plotkey_ratio( ax, gdict, K, fx, xkey, ykey, cond1, cond2, dat, base, **kwargs):
    try:
        defmarker = gdict['marker']
    except:
        defmarker = 'o'  
    marker = kwargs.get( 'marker', defmarker)
    ms     = kwargs.get( 'ms', 5. ) 
    mec    = kwargs.get( 'mec', gdict['ec'])
    mfc    = kwargs.get( 'mfc', gdict['fc'])
    mew    = kwargs.get( 'mew', 1.0 ) 
    ew     = kwargs.get( 'ew', 1.0 )
    ecap   = kwargs.get( 'ecap', 0. )
    save   = kwargs.get( 'save', False ) 
    labelstr = kwargs.get( 'labelstr', gdict['label'] ) 
  
    xoffset = kwargs.get( 'xoffset', 0.) 
    exceptions = kwargs.get( 'exceptions', False)

    operation = kwargs.get( 'operation', 'ratio') 
  
    try:
        xc, rval, rerr, nsamples = data_ratio( dat, cond1, cond2, K, xkey, ykey, \
                                       operation=operation )
        rval = rval / base 
        rerr = rerr / base 
        #print '\t%s: x points in common = '%ykey,len(xc)
    
        yf = kwargs.get( 'yf', None)
        if yf is not None: 
            ydict = kwargs.get('yf_kwargs', {} )  
            yf_usex = kwargs.get('yf_usex', False)
            if yf_usex is True:
                ydict['x'] = fx(xc)
            runc = unumpy.uarray(( rval, rerr ))
            runc = yf( runc, **ydict ) 
            rval = unumpy.nominal_values( runc  ) 
            rerr = unumpy.std_devs( runc )

    
        if len(xc) > 0:
            ## Debugging 
            #print "plotkey_ratio debug"
            #print "len( x ) = ", len(fx(xc)) 
            #print "len( y ) = ", len(rval/base)  
            #print "len( yerr ) = ", len(rerr/base)  
            ax.errorbar( fx(xc) + xoffset, rval/base, yerr=rerr/base,\
                      capsize=ecap, elinewidth=ew,\
                      fmt='.', ecolor=mec, mec=mec,\
                      mew=mew, marker=marker, mfc=mfc, ms=ms,\
                      label=labelstr)
            ax_relerr = kwargs.get('ax_relerr', None)
            if ax_relerr is not None:
                mec_relerr    = kwargs.get('mec_relerr', mec ) 
                mew_relerr    = kwargs.get('mec_relerr', mew ) 
                marker_relerr = kwargs.get('mec_relerr', marker ) 
                mfc_relerr    = kwargs.get('mfc_relerr', mfc )
                ms_relerr     = kwargs.get('ms_relerr', ms )
 
                ax_relerr.plot( fx(xc), 100.*2.*rerr/rval, '.',\
                      mec=mec_relerr, mew=mew_relerr, \
                      marker=marker_relerr, mfc=mfc_relerr, \
                      ms=ms_relerr,\
                      label=labelstr)
              
        if save:
            fname = xkey.replace(':','_') + '_' +  labelstr[-3:] + '.rawdatNOTCorrected' 
            X =  np.transpose(np.vstack( ( fx(xc), rval/base, rerr/base )))
            np.savetxt( fname, X, fmt='%10.2f', delimiter='\t', newline='\n')
        if kwargs.get('return_unumpy', False): 
            return {'x':fx(xc), 'y':unumpy.uarray(( rval/base, rerr/base)),\
                    'nsamples':nsamples }
        else:
            return [fx(xc), rval/base, rerr/base, nsamples]
    except:
        print "Exception occured in plotkey_ratio %s vs %s"%(ykey,xkey)
        if exceptions:
            raise

