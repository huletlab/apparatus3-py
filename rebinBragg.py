
  print "Rebin %s" % shotnum
  #Cropped and binned data is saved in separate dirs
  binszs = (1, 2, 3, 4, 5, 6, 8)
  for b in binszs:
      path = args.path.split(os.sep)[0] + '_rebin_%d' % b
      if not os.path.exists(path):
          print "...Creating directory %s" % path
          os.makedirs(path)
      savepath = os.path.join( path , shot + '_bragg.dat')
      #print savepath

      row, col = reout.shape[0]/2, reout.shape[1]/2
      
      pngprefix = os.path.join(path, shotnum + '_bragg')
      falsecolor.inspecpng( [reout], \
                            row, col, reout.min(), reout.max(), \
                            falsecolor.my_grayscale, \
                            pngprefix, 100, origin = 'upper' , \
                            step=True, scale=10, \
                            interpolation='nearest', \
                            extratext='')
