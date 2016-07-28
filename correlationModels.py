'''
Compare three isotropic and homogeneous correlation models:

-   Foar (first order auto-regressive model)
-   Soar (second order auto-regressive model)
-   Gaussian

It plots the correlations as a function of distance and their power spectra.
It also generates and plots one random spatial realization of each of these models.

Worth mentioning, the correlation length is defined by the distance where the correlation reaches 1/sqrt(e).

Correlations models are instances of the `CorrModel` (itself deriving from `Covariance` class) and are defined in `./DM93/covarianceCls.py`.
All `Covariance` instances provide their matrix representation and a random generator.
`CorrModel` instances provide as well their radial correlation function, and their theoretical power spectrum (as obtained on an infinite domain).
'''
import numpy as np 
import matplotlib.pyplot as plt
from DM93 import Uncorrelated, Foar, Soar, Gaussian

#====================================================================
#===| setup and configuration |======================================

execfile('config.py')

Lc = grid.L/20.


corrModels = {  'foar' : Foar(grid, Lc),
                'soar' : Soar(grid, Lc),
                'gaussian' : Gaussian(grid, Lc),
                }


# -- generating random realizations of correlated signal

realizations = dict()
for label, cm in corrModels.iteritems():
    realizations[label] = cm.random()

#====================================================================
#===| plots |========================================================

# -- correlation models and spectra

fig1 = plt.figure()
fig1.subplots_adjust(hspace=0.6)
for label, cm in corrModels.iteritems():
    axGrid = plt.subplot(311)
    axSpTh = plt.subplot(312)
    axReal = plt.subplot(313)
    axGrid.plot(grid.x, cm.corrFunc(), label=label)
    axSpTh.plot(grid.halfK, cm.powSpecTh(), label=label)
    axReal.plot(grid.x, realizations[label], label=label)

axGrid.set_title('Correlation $L_c=%.0f$ km'%(Lc/km))

xticklabels, xticks = grid.ticks(units=km)[:2]
axGrid.set_xticks(xticks)
axGrid.set_xticklabels(xticklabels)
axGrid.set_xlabel('distance [km]')

axSpTh.set_title('Normalized theoretical power spectrum')
axSpTh.set_xlabel('wavenumber $k$')


axReal.set_title('Random realization')
axReal.set_xlabel('$x$ [km]')
axReal.set_xticks(xticks)
axReal.set_xticklabels(xticklabels)

axSpTh.legend(loc='best')


plt.show()
