import numpy as np
import matplotlib.pyplot as plt

Xps = np.array([ np.loadtxt("xp_%d.dat" % x ) for x in range(200) ]) 
Lambdas = np.loadtxt("lambdas.dat")
plt.plot(Lambdas,Xps)
plt.show()
