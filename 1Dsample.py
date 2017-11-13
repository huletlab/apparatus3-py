#!/usr/bin/python
from oneDsample_lib import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--range', action="store", dest='range', help="analyze files in the specified range.")
    parser.add_argument('--fname', action="store", dest='fname', help="Filename",default= "")
    parser.add_argument('--folder', action="store", dest='folder', help="Folder",default= "plots")
    args = parser.parse_args()
    datadir = "./"
    fname = "{0}_1D_sample.png".format(args.fname or args.range)
    shots = qrange.parse_range(args.range)
    process(shots, fname=fname,folder = args.folder,save_txt=True)
