import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def statimage( Z, rois ):

   fig = plt.figure(figsize=(16.,7.5))

   rows = len(rois)
   ax = plt.subplot2grid( (rows, 4), (0,0), rowspan=rows, colspan=2)

   colormap = matplotlib.cm.spectral

   im = ax.imshow( Z, \
                    interpolation='bilinear', \
                    cmap=colormap, \
                    origin='lower',\
                    vmax= Z.max(), \
                    vmin= Z.min())

   cbar = fig.colorbar( im )

   

   dat0=None
   cmin=Z.min()
   cmaz=Z.max()
   means = []
   stdevs = []

   for i,r in enumerate(rois):
     rect = matplotlib.patches.Rectangle( (r[0],r[1]), r[2], r[3], fill=False, ec="black") 
     ax.text( r[0]+2,r[1]+2, '%d' % i ) 
     ax.add_patch(rect) 
     axr = plt.subplot2grid( (rows,4), (i,2))
     rectdat =  Z[ r[0]:r[0]+r[2], r[1]:r[1]+r[3] ]
     if i == 0:
        dat0 = rectdat
        cmin = rectdat.min()
        cmax = rectdat.max()
     axr.imshow( rectdat, \
                    interpolation='nearest', \
                    cmap=colormap, \
                    origin='lower',\
                    vmax= cmax, \
                    vmin= cmin)

     axh = plt.subplot2grid( (rows,4), (i,3))
     n, bins, patches = axh.hist( rectdat.reshape( rectdat.shape[0]*rectdat.shape[1]), 80,\
                                  range = (Z.min(), Z.max() ), log=False,\
                                  normed=False, facecolor='red' if i==0 else 'green', alpha=1.0)
     n, bins, patches = axh.hist( dat0.reshape( dat0.shape[0]*dat0.shape[1]), 80,\
                                  range = (Z.min(), Z.max() ), log=False,\
                                  normed=False, facecolor='red', alpha=0.4)

     means.append( np.mean(rectdat) )
     stdevs.append( np.std(rectdat) ) 
     axh.text(0.95, 0.95,' mean = %4.2f\nstdev = %4.2f' % ( means[-1], stdevs[-1] ),
     horizontalalignment='right',
     verticalalignment='top',
     size=10,
     transform = axh.transAxes)

     
   print means
   print stdevs
   plt.show() 
   return means, stdevs

def maskedimages( imgs, savepath ):


   rows = 2*len(imgs)  
   cols = 4
   
   fig = plt.figure(figsize=(12.,4*len(imgs)))

   for i,img in enumerate(imgs):
      if img != None:

         ax = plt.subplot2grid( (rows, cols), (2*i,0), rowspan=2, colspan=2)

         ax.set_title( img[1]['camera'] )

         colormap = matplotlib.cm.spectral

         im = ax.imshow( img[0], \
                    interpolation='nearest', \
                    cmap=colormap, \
                    origin='lower',\
                    vmax= img[0].max(), \
                    vmin= img[0].min())

         #cbar = fig.colorbar( im )
         plt.colorbar(im, use_gridspec=True)

         sr = img[1]['signalregion'] 
         rect = matplotlib.patches.Rectangle( \
                (sr[0]-0.5,sr[2]-0.5), sr[1]-sr[0], sr[3]-sr[2], \
                fill=False, ec="black") 
         ax.add_patch(rect) 


         # Plot signal

         #signal = img[0][ img[2][0]:img[2][1], img[2][2]:img[2][3] ]
         signal = img[1]['signalpx'] 
         ax = plt.subplot2grid( (rows,cols), (2*i,2), rowspan=1, colspan=1)
         im = ax.imshow( signal, \
                    interpolation='nearest', \
                    cmap=colormap, \
                    origin='lower',\
                    vmax= img[0].max(), \
                    vmin= img[0].min())


         axh1 = plt.subplot2grid( (rows,cols), (2*i,3), rowspan=1, colspan=1)
         n, bins, patches = axh1.hist( signal.reshape( signal.shape[0]*signal.shape[1]), 80,\
                                  range = (img[0].min(), img[0].max() ), log=False,\
                                  normed=False, facecolor='green', alpha=1.0)


         # Plot background 

         #masked = img[0]
         #mask = np.zeros_like(masked)
         #mask[ img[2][0]:img[2][1], img[2][2]:img[2][3] ] = 1
         #masked = np.ma.MaskedArray(masked, mask= mask)
         masked = img[1]['maskedpx'] 
    
         maskedpixels =  masked.compressed()

         ax = plt.subplot2grid( (rows,cols), (2*i+1,2), rowspan=1, colspan=1)
         im = ax.imshow( masked, \
                    interpolation='nearest', \
                    cmap=colormap, \
                    origin='lower',\
                    vmax= img[0].max(), \
                    vmin= img[0].min())
          
         axh2 = plt.subplot2grid( (rows,cols), (2*i+1,3), rowspan=1, colspan=1)
         n, bins, patches = axh2.hist( maskedpixels, 80,\
                                  range = (img[0].min(), img[0].max() ), log=False,\
                                  normed=False, facecolor='red', alpha=1.0)

         mean_sig = np.mean(signal)
         stdev_sig = np.std(signal)

         mean_bgnd = np.mean(maskedpixels)
         stdev_bgnd = np.std(maskedpixels)

         npix = signal.size
         netsig = npix*(mean_sig - mean_bgnd)

         sigerror = np.sqrt(npix)*stdev_bgnd

         axh1.text( 0.95,0.95, \
                   'mean = %4.2f' % mean_sig +\
                   '\nstdev = %4.2f' % stdev_sig +\
                   '\nN pixels = %d' % npix +\
                   '\nsignal = %4.2f' % netsig +\
                   '\n+/- %3.2f' % sigerror,\
                   horizontalalignment='right', \
                   verticalalignment='top', \
                   size=8,\
                   transform = axh1.transAxes) 

         axh2.text( 0.95,0.95, ' mean = %4.2f\nstdev = %4.2f' % (mean_bgnd, stdev_bgnd), \
                   horizontalalignment='right', \
                   verticalalignment='top', \
                   size=8,\
                   transform = axh2.transAxes) 
  
  
   plt.tight_layout()
   plt.savefig( savepath )  
   return 
      

def maskedimages_all( imgs, savepath ):


   rows = 2*len(imgs)  
   cols = 10
   
   fig = plt.figure(figsize=(20.,4*len(imgs)))

   for i,img in enumerate(imgs):
      if img != None:

         ax = plt.subplot2grid( (rows, cols), (2*i,0), rowspan=2, colspan=2)
         ax.set_title( img[1]['camera'] + ' atoms')
         colormap = matplotlib.cm.spectral
         im = ax.imshow( img[2], \
                    interpolation='nearest', \
                    cmap=colormap, \
                    origin='lower',\
                    vmax= img[3].max(), \
                    vmin= img[3].min())
         plt.colorbar(im, use_gridspec=True)

         ax = plt.subplot2grid( (rows, cols), (2*i,2), rowspan=2, colspan=2)
         ax.set_title( img[1]['camera'] + ' noatoms')
         colormap = matplotlib.cm.spectral
         im = ax.imshow( img[3], \
                    interpolation='nearest', \
                    cmap=colormap, \
                    origin='lower',\
                    vmax= img[3].max(), \
                    vmin= img[3].min())
         plt.colorbar(im, use_gridspec=True)

         ax = plt.subplot2grid( (rows, cols), (2*i,4), rowspan=2, colspan=2)
         ax.set_title( img[1]['camera'] + ' diff')
         colormap = matplotlib.cm.spectral
         im = ax.imshow( img[2]-img[3], \
                    interpolation='nearest', \
                    cmap=colormap, \
                    origin='lower',\
                    vmax= (img[2]-img[3]).max(), \
                    vmin= (img[2]-img[3]).min())
         plt.colorbar(im, use_gridspec=True)

         ax = plt.subplot2grid( (rows, cols), (2*i,6), rowspan=2, colspan=2)
         ax.set_title( img[1]['camera'] )
         colormap = matplotlib.cm.spectral
         im = ax.imshow( img[0], \
                    interpolation='nearest', \
                    cmap=colormap, \
                    origin='lower',\
                    vmax= img[0].max(), \
                    vmin= img[0].min())
         plt.colorbar(im, use_gridspec=True)

         sr = img[1]['signalregion'] 
         rect = matplotlib.patches.Rectangle( \
                (sr[0]-0.5,sr[2]-0.5), sr[1]-sr[0], sr[3]-sr[2], \
                fill=False, ec="black") 
         ax.add_patch(rect) 


         # Plot signal

         #signal = img[0][ img[2][0]:img[2][1], img[2][2]:img[2][3] ]
         signal = img[1]['signalpx'] 
         ax = plt.subplot2grid( (rows,cols), (2*i,8), rowspan=1, colspan=1)
         im = ax.imshow( signal, \
                    interpolation='nearest', \
                    cmap=colormap, \
                    origin='lower',\
                    vmax= img[0].max(), \
                    vmin= img[0].min())


         axh1 = plt.subplot2grid( (rows,cols), (2*i,9), rowspan=1, colspan=1)
         n, bins, patches = axh1.hist( signal.reshape( signal.shape[0]*signal.shape[1]), 80,\
                                  range = (img[0].min(), img[0].max() ), log=False,\
                                  normed=False, facecolor='green', alpha=1.0)


         # Plot background 

         #masked = img[0]
         #mask = np.zeros_like(masked)
         #mask[ img[2][0]:img[2][1], img[2][2]:img[2][3] ] = 1
         #masked = np.ma.MaskedArray(masked, mask= mask)
         masked = img[1]['maskedpx'] 
    
         maskedpixels =  masked.compressed()

         ax = plt.subplot2grid( (rows,cols), (2*i+1,8), rowspan=1, colspan=1)
         im = ax.imshow( masked, \
                    interpolation='nearest', \
                    cmap=colormap, \
                    origin='lower',\
                    vmax= img[0].max(), \
                    vmin= img[0].min())
          
         axh2 = plt.subplot2grid( (rows,cols), (2*i+1,9), rowspan=1, colspan=1)
         n, bins, patches = axh2.hist( maskedpixels, 80,\
                                  range = (img[0].min(), img[0].max() ), log=False,\
                                  normed=False, facecolor='red', alpha=1.0)

         mean_sig = np.mean(signal)
         stdev_sig = np.std(signal)

         mean_bgnd = np.mean(maskedpixels)
         stdev_bgnd = np.std(maskedpixels)

         npix = signal.size
         netsig = npix*(mean_sig - mean_bgnd)

         sigerror = np.sqrt(npix)*stdev_bgnd

         axh1.text( 0.95,0.95, \
                   'mean = %4.2f' % mean_sig +\
                   '\nstdev = %4.2f' % stdev_sig +\
                   '\nN pixels = %d' % npix +\
                   '\nsignal = %4.2f' % netsig +\
                   '\n+/- %3.2f' % sigerror,\
                   horizontalalignment='right', \
                   verticalalignment='top', \
                   size=8,\
                   transform = axh1.transAxes) 

         axh2.text( 0.95,0.95, ' mean = %4.2f\nstdev = %4.2f' % (mean_bgnd, stdev_bgnd), \
                   horizontalalignment='right', \
                   verticalalignment='top', \
                   size=8,\
                   transform = axh2.transAxes) 
  
  
   plt.tight_layout()
   plt.savefig( savepath, dpi=120 )  
   return 
      

