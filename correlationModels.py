import numpy as np 
import matplotlib.pyplot as plt
from DM93 import Grid, Uncorrelated, Foar, Soar, Gaussian

#====================================================================
#===| setup and configuration |======================================

# -- units of space: m 
km = 1000.

# -- discretization
a = 2500.*km
L = 2.*np.pi * a
N = 24
grid = Grid(N, L)

Lc = a/4.


corrModels = {  'uncorrelated' : Uncorrelated(grid),
                'foar' : Foar(grid, Lc),
                'soar' : Soar(grid, Lc),
                'gaussian' : Gaussian(grid, Lc),
                }


#====================================================================
#===| plots |========================================================

fig = plt.figure(figsize=(8,8))
fig.subplots_adjust(hspace=0.6)
for label, cm in corrModels.iteritems():
    axGrid = plt.subplot(311)
    axSpTh = plt.subplot(312)
    axSpNu = plt.subplot(313)
    axGrid.plot(grid.x, cm.corrFunc(), label=label)
    axSpTh.plot(grid.halfK, cm.powSpecTh(), label=label)

    axSpNu.plot(grid.halfK, cm.powSpecNum(), label=label)

axGrid.set_title('Correlation $L_c=%.1e$ m'%Lc)

xticklabels, xticks = grid.ticks(3, units=km)[:2]
axGrid.set_xticks(xticks)
axGrid.set_xticklabels(xticklabels)
axGrid.set_xlabel('distance [km]')

axSpTh.set_title('Normalized theoretical power spectrum')
axSpTh.set_xlabel('wavenumber $k$')

axSpNu.set_title('Normalized numerical power spectrum')
axSpNu.set_xlabel('wavenumber $k$')

axSpNu.legend(loc='best')

plt.show()
