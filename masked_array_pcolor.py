import numpy as np
import matplotlib.pyplot as plt

cmap = plt.get_cmap("jet")
cmap.set_bad(color="gray",alpha=1.)
C = np.ma.masked_invalid(C)
plt.figure()
plt.pcolor(C,cmap=cmap)

import numpy as np
import matplotlib.pyplot as plt
cmap = plt.get_cmap("jet")
cmap.set_bad(color="gray",alpha=1.)
C_nn = np.ma.masked_invalid(C_nn)
plt.figure()
plt.pcolormesh(C_nn,cmap=cmap)
plt.xlabel("Residue i")
plt.ylabel("Residue j")
cbar = plt.colorbar()
cbar.set_label("Interaction strength")
plt.title("1r69 random non-native  $\\overline{\\epsilon_{nn}} = 0$  $\\sigma_{\\epsilon_{nn}}^2 = 1$ ")


