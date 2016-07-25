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
nuFactors  = [0, .001, 0.01, 0.05]

# -- Correlations
corrObs = Uncorrelated(grid)

Lc = a/6.
corrMod = Soar(grid, Lc)

# -- correlation power spectra
r2 = corrObs.powSpec()
q2 = corrMod.powSpec()

#====================================================================
#===| computations |=================================================

f2Plus = dict()
cPlus = dict()

for nuF in nuFactors:
    nu =  nuF/dt*a**2
    # -- assymptotic variances spectra (forecast and analysis respectively)
    f2Plus[nuF] = spVarStationary(grid, r2, q2, dt=dt, nu=nu)[0]

    # -- assymptotic convergence rate spectrum
    cPlus[nuF] = convRateAssymp(grid, r2, q2, dt=dt, nu=nu)


#====================================================================
#===| plots |========================================================

fig = plt.figure()
axVar = plt.subplot(211)
axConv = plt.subplot(212)


nuFStr = r'$\frac{\nu\Delta t}{a^2}=$'
for nuF in nuFactors:
    axVar.plot(grid.halfK, f2Plus[nuF], label='%s %.3f'%(nuFStr, nuF))
    axConv.plot(grid.halfK, cPlus[nuF], label='%s %.3f'%(nuFStr, nuF))

axVar.set_yscale('log')
axConv.set_yscale('log')

axVar.set_xticks(())
axConv.set_xlabel('wavenumber $k$')

axVar.set_title('Assymptotical variance spectra')
axConv.set_title('Assymptotical convergence spectra')
axConv.legend(loc='best')

plt.show()
