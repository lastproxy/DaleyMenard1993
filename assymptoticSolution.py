import numpy as np 
import matplotlib.pyplot as plt

from DM93 import Grid, Uncorrelated, Foar, Soar, Gaussian
from DM93 import spVarStationary, analSpVar, convRateAssymp

#====================================================================
#===| setup and configuration |======================================

# -- units of space: m and time: s
km = 1000.
h = 3600.

# -- discretization
a = 2500.*km
L = 2.*np.pi * a
N = 24
grid = Grid(N, L)
dt =6.*h

# -- viscosity
nuFactor = 0.
nu =  nuFactor/dt*a**2

# -- Correlations
corrObs = Uncorrelated(grid)

Lc = a/6.
corrMod = Soar(grid, Lc)

# -- correlation power spectra
r2 = corrObs.powSpec()
q2 = corrMod.powSpec()

#====================================================================
#===| computations |=================================================

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

axe.set_yscale('log')
axe.set_xlabel('wavenumber $k$')
axe.set_title('Assymptotical variance and convergence spectra')
axe.legend(loc='best')

plt.show()
