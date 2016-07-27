import numpy as np 
import matplotlib.pyplot as plt

from DM93 import Uncorrelated, Foar, Soar, Gaussian
from DM93 import analSpVar

#====================================================================
#===| setup and configuration |======================================

execfile('config.py')

# -- observation errors
obsLc = None
obsCorr = Uncorrelated(grid, obsLc)

# -- forecast errors
fctLc = grid.L/50.
fctCorr = Soar(grid, fctLc)

#====================================================================
#===| computations |=================================================

# -- correlation power spectra
r2 = obsCorr.powSpecTh()
f2 = fctCorr.powSpecTh()

a2 = analSpVar(f2, r2)

#====================================================================
#===| plots |========================================================

axe = plt.subplot(111)

axe.plot(grid.halfK, f2, label=r'$f^2$ (%s, $L_c=%d$ km)'%(fctCorr.name, fctCorr.Lc/km))
axe.plot(grid.halfK, r2, label=r'$r^2$ (%s, $L_c=%d$ km)'%(obsCorr.name, obsCorr.Lc/km))
axe.plot(grid.halfK, a2, label=r'$a^2$')

axe.set_yscale('log')
axe.set_xlabel('wavenumber $k$')
axe.set_title('Variances spectra')
axe.legend(loc='best')


plt.show()
