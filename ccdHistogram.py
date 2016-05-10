import braggEigen
import os
import numpy as np
import matplotlib


def histo( imgs , shot='histo', savepath ='' ):
    imgs = np.array ( imgs ) 
     
    figure = matplotlib.figure.Figure(figsize=(6.,5.25) ) 
    gs = matplotlib.gridspec.GridSpec( 1,1) 
    ax = figure.add_subplot( gs[0,0] )

    nbins = int(np.amax(imgs.flat) - np.amin( imgs.flat )  + 1)
    
    ax.hist( imgs.flat , nbins , histtype='stepfilled' )
    ax.set_xlabel('CCD counts')
    outfile = shot + '_bgshisto.png' 
    ax.set_title( outfile  ) 
    gs.tight_layout(figure, rect=[0,0.0,1.,0.95])
    figure.savefig(savepath + outfile, dpi=250)

def phisto( ax, dat, title='', xl='' ):
    nbins = int( np.amax(dat) - np.amin(dat) + 1 ) 
    ax.hist( dat.flat, nbins, histtype='stepfilled') 
    ax.set_title( title ) 
    ax.set_xlabel( xl ) 
    txt = 'mean=%.3g\nstdev=%.3g' % (np.mean(dat), np.std(dat)) 
    ax.text(0.98,0.98, txt, transform=ax.transAxes, fontsize=10, va='top', ha='right',\
            bbox=dict(boxstyle='round', facecolor='white') )
    ax.grid()
    

def histoeigen( imgs , atoms, eigenbg, eigen,  shot='histo', savepath ='' ):
    outfile = shot + '_eigenhisto.png' 
    imgs = np.array ( imgs ) 
     
    figure = matplotlib.figure.Figure(figsize=(12.,10.5) ) 
    gs = matplotlib.gridspec.GridSpec( 2,2)
 
    axbgs = figure.add_subplot( gs[0,0] )
    phisto( axbgs, imgs, title='avg of bgnds histo',\
                         xl='CCD counts') 

    axatoms = figure.add_subplot( gs[0,1] )
    phisto( axatoms, atoms, title='atoms histo',\
                            xl='CCD counts')
 
    axeigenbg = figure.add_subplot( gs[1,1] )
    phisto( axeigenbg, eigenbg, title='eigenbg histo',\
                                xl='CCD counts') 

    axeigen = figure.add_subplot( gs[1,0] )
    phisto( axeigen, eigen, title='atoms-eigenbg histo',\
                            xl='counts')

    gs.tight_layout(figure, rect=[0,0.0,1.,0.95])
    figure.savefig(savepath + outfile, dpi=250)


