""" Calculate the autocorrelation function using fast fourier transform

"""

from numpy.fft import fft,ifft
import numpy as np


def acf(x):

    N = float(len(x))
    pow2 = int(2**np.ceil(np.log2(len(x))))
    x_new = np.zeros(pow2,float)
    x_new[:len(x)] = x

    FT = fft.fft(x_new)
    acf = (fft.ifft(FT*conjugate(FT)).real)/N
    return acf

x = np.loadtxt("skip30.ev")
dc1 = x[:,1]/x[:,0]
N = float(len(dc1))

pow2 = int(2**np.ceil(np.log2(len(dc1))))
x = np.zeros(pow2,float)
x[:len(dc1)] = dc1

FT = fft(x)
acf = (ifft(FT*np.conjugate(FT)).real)/N

np.savetxt("acf",acf)
