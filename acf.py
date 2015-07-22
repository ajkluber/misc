""" Calculate the autocorrelation function using fast fourier transform

"""

from numpy.fft import fft,ifft
import numpy as np

def calculate_acf(x):
    from numpy.fft import fft,ifft

    N = float(len(x))
    pow2 = int(2**np.ceil(np.log2(len(x))))
    x_new = np.zeros(pow2,float)
    x_new[:len(x)] = x

    FT = fft(x_new)
    acf = (ifft(FT*conjugate(FT)).real)/N
    acf /= acf.max()
    acf = acf[:len(acf)/2]
    return acf

if __name__ == "__main__":
    x = np.loadtxt("skip30.ev")
    dc1 = x[:,1]/x[:,0]

    pow2 = int(2**np.ceil(np.log2(len(dc1))))
    x = np.zeros(pow2,float)
    x[:len(dc1)] = dc1

    from numpy.fft import fft,ifft

    pow2 = int(2**np.ceil(np.log2(len(x))))
    x2 = np.zeros(pow2,float)
    x2[:len(x)] = x - x.mean()
    N = float(len(x2))
    FT = fft(x2)
    acf_raw = (ifft(FT*np.conjugate(FT)).real)/N
    acf = acf_raw[:len(acf_raw)/2]/acf_raw[0]


    np.savetxt("acf",acf)


    delta_s = 100
    s_max = 4000
    freq = np.logspace(-5,-1,200)
    Laplace = np.array([ np.dot((freq[i + 1] - freq[i])*np.exp(-freq[i]*np.arange(len(acf)/2)),acf[:len(acf)/2]) for i in range(len(freq)-1) ])
