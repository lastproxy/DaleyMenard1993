import numpy as np 
import matplotlib.pyplot as plt

from numpy import pi 
from DM93 import Uncorrelated, Foar, Soar, Gaussian
from DM93 import spVarStationary, analSpVar, convRateAssymp

#====================================================================
#===| setup and configuration |======================================

execfile('config.py')

# -- viscosity
nuFactors  = [0, .0001, .001,]

# -- Correlations
obsCorr = Uncorrelated(grid)

fctLc = grid.L/20.
fctCorr = Soar(grid, fctLc)

#====================================================================
#===| computations |=================================================

# -- correlation power spectra
r2 = obsCorr.powSpecTh()
q2 = fctCorr.powSpecTh()


f2Plus = dict()
cPlus = dict()

for nuF in nuFactors:
    nu =  nuF/dt*(2.*pi*grid.L)**2
    # -- assymptotic variances spectra (forecast and analysis respectively)
    f2Plus[nuF] = spVarStationary(grid, r2, q2, dt=dt, nu=nu)[0]

    # -- assymptotic convergence rate spectrum
    cPlus[nuF] = convRateAssymp(grid, r2, q2, dt=dt, nu=nu)


#====================================================================
#===| plots |========================================================

fig = plt.figure()
axVar = plt.subplot(211)
axConv = plt.subplot(212)


nuFStr = r'$\frac{4\pi^2\nu\Delta t}{L^2}=$'
for nuF in nuFactors:
    axVar.plot(grid.halfK, f2Plus[nuF], label='%s %.3f'%(nuFStr, nuF))
    axConv.plot(grid.halfK, cPlus[nuF], label='%s %.3f'%(nuFStr, nuF))

axVar.set_xscale('log')
axVar.set_yscale('log')
axConv.set_xscale('log')
axConv.set_yscale('log')

axVar.set_xticks(())
axConv.set_xlabel('wavenumber $k$')

axVar.set_title('Assymptotical variance spectra')
axConv.set_title('Assymptotical convergence spectra')
axConv.legend(loc='best')

plt.show()
