import numpy as np 
import matplotlib.pyplot as plt

from DM93 import Uncorrelated, Foar, Soar, Gaussian
from DM93 import spVarStationary, analSpVar, convRateAssymp

#====================================================================
#===| setup and configuration |======================================

execfile('config.py')

# -- Correlations
obsLc = None
obsCorr = Uncorrelated(grid, obsLc)

modLc = grid.L/20.
modCorr = Soar(grid, modLc)

#====================================================================
#===| computations |=================================================

# -- correlation power spectra
r2 = obsCorr.powSpecTh()
q2 = modCorr.powSpecTh()

# -- assymptotic variances spectra (forecast and analysis respectively)
f2Plus = spVarStationary(grid, r2, q2, dt=dt, nu=nu)[0]
analPlus = analSpVar(f2Plus, r2, q2)

# -- assymptotic convergence rate spectrum
cPlus = convRateAssymp(grid, r2, q2, dt=dt, nu=nu)


#====================================================================
#===| plots |========================================================

fig = plt.figure()
axe = plt.subplot(111)

axe.plot(grid.halfK, f2Plus, label=r'$\overline{f}_+^2$')
axe.plot(grid.halfK, analPlus, label=r'$\overline{a}_+^2$')
axe.plot(grid.halfK, cPlus, label=r'$\overline{c}_+$')
axe.plot(grid.halfK, r2, label=r'$r^2$')

axe.set_yscale('log')
axe.set_xlabel('wavenumber $k$')
axe.set_title('Assymptotical variance and convergence spectra')
axe.legend(loc='best')

plt.show()
