import numpy as np 
from gridCls import Grid
import matplotlib.pyplot as plt
from corrModelLib import Uncorrelated, Foar, Soar, Gaussian
from DM93PlotLib import (   plotCorrPowSpectra, plotImageGF2, 
                            plotAssympVar,  plotAssympConvRate, 
                            plotViscAssympVar, plotViscAssympConv
                            )

# -- Grid
# units of space = m and time = s
a = 2500. * 1000.   # 2500 km
L = 2.*np.pi * a
N = 24
grid = Grid(N, L)
dt =6. * 3600.      # 6 hours
# -- viscosity
nuFactor = 0.
nu =  nuFactor/dt*a**2

# -- Correlations
corrObs = Uncorrelated(grid)

lMod = a/6.
corrMod = Soar(grid, lMod)

# -- correlation power spectra
r2 = corrObs.powSpec()
q2 = corrMod.powSpec()


plt.figure()
axe = plotCorrPowSpectra(grid, r2, q2)

# -- convergence
nIter = 5
plt.figure()
f20=-0.01
k = 18
axe = plotImageGF2( grid, k, r2, q2, f20=f20, dt=dt, nu=nu, nIter=nIter)

# -- assymptotical variances spectrum
plt.figure()
axe = plotAssympVar(grid, r2, q2, dt=dt, nu=nu, axe=plt.subplot(111))
# -- assymptotical convergence spectrum
axe = plotAssympConvRate(grid, r2, q2, dt=dt, nu=nu, axe=axe)
axe.set_title('Assymptotical variances and convergence rate')





# -- viscosity impact
nuFactorList  = np.array([-0.01, -0.001, 0, .001, 0.01])
nuList  =  nuFactorList/dt*a**2
plt.figure()
ax1 = plotViscAssympVar(grid, nuList, r2, q2, dt=dt, axe=plt.subplot(211), 
                        legend=False)
ax1.set_xlabel('')
ax1.set_xticks(())
ax2 = plotViscAssympConv(   grid, nuList, r2, q2, dt=dt, axe=plt.subplot(212),
                            legend=False)
ax2.legend(loc='upper left')
