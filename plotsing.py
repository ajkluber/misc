import numpy as np
import matplotlib.pyplot as plt

s = np.loadtxt("singular_values.dat")
n_vals = len(s)


s2 = s**2
y = s2/sum(s2)
plt.plot(y,'r.')
for tau in [0.001,0.01,0.1]:
    plt.axhline(y=tau,xmin=0,xmax=n_vals,color='k')
plt.xlabel("Singular value index")
plt.ylabel("$\\sigma_i^2$/sum($\\sigma_i^2$)")

#s_nrm = s/max(s)
#taus = [0.001,0.01,0.1,0.5]
#taus = np.linspace(min(s),max(s),5)
#taus = np.logspace(np.log10(min(s)),np.log10(max(s)),5)
#for tau in taus:
#    plt.plot((s**2)/((s**2) + (tau**2)),lw=2,label=str(tau))
#plt.legend(loc=2)
#plt.xlabel("Singular value index")

plt.show()
