import sys, math
sys.path.append('L:/software/apparatus3/imagesgui')

from scipy import *

from import_data import *


def find_( a ):
    a0=a[511]
    for i in reversed(range(512)):
        if a[i] > a0+12:
            return i
        a0=a[i]
    return 0


if __name__ == '__main__':

    dir='L:/data/app3/2011/1109/110907/'

    m740 = load_fits_file(dir+'mag740.fits')[:,1]
    m750 = load_fits_file(dir+'mag750.fits')[:,1]
    m760 = load_fits_file(dir+'mag760.fits')[:,1]
    m770 = load_fits_file(dir+'mag770.fits')[:,1]
    m780 = load_fits_file(dir+'mag780.fits')[:,1]
    m800 = load_fits_file(dir+'mag800.fits')[:,1]
    m820 = load_fits_file(dir+'mag820.fits')[:,1]
    m840 = load_fits_file(dir+'mag840.fits')[:,1]
    m860 = load_fits_file(dir+'mag860.fits')[:,1]

    #matshow( m740, cmap=cm.gist_earth_r )
    x=linspace(0,511,512)

    #plt.plot (x,m740, x, m750, x, m760, x, m770, x, m780, x, m800, x, m820, x, m840, x, m860)
    
    stepsX=array([740,750,760,770,780,800,820,840,860])
    stepsY=array([find_(m740), find_(m750), find_(m760), find_(m770), find_(m780), find_(m800), find_(m820), find_(m840), find_(m860)])

    (ar,br) = polyfit(stepsX,stepsY,1)
    Yfit = polyval([ar,br],stepsX) 

    print('fit results: a=%.2f b=%.2f' % (ar,br))
    plt.title('')
    plt.plot(stepsX,stepsY,'o')
    plt.plot(stepsX,Yfit,'-')
    show()
    
    
    