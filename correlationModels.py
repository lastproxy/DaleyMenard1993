import numpy as np 
import matplotlib.pyplot as plt
from DM93 import Grid, Uncorrelated, Foar, Soar, Gaussian,

# -- grid definition
#       units of space = m and time = s

km = 1000.
a = 2500. * km 
L = 2.*np.pi * a
N = 48
grid = Grid(N, L)

Lc = a/4.


corrModels = {  'uncorrelated' : Uncorrelated(grid),
                'foar' : Foar(grid, Lc),
                'soar' : Soar(grid, Lc),
                'gaussian' : Gaussian(grid, Lc),
                }


fig = plt.figure()
fig.subplots_adjust(hspace=0.4)
for label, cm in corrModels.iteritems():
    axGrid = plt.subplot(211)
    axSpec = plt.subplot(212)
    axGrid.plot(grid.x, cm.corrFunc(), label=label)
    axSpec.plot(grid.k, cm.powSpec(), label=label)

axGrid.set_title('Correlation $L_c=%.1e$ m'%Lc)
axGrid.set_xticks(np.floor([-grid.L/4., 0., grid.L/4.]))
axGrid.set_xlabel('distance [m]')

axSpec.set_title('Normalized power spectrum')
axSpec.set_xlabel('wavenumber $k$')

axSpec.legend(loc='best')
