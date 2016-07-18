import numpy as np 
from gridCls import Grid
from corrModelLib import Uncorrelated, Soar
from DM93PlotLib import plotCorrPowSpectra, plotImageGF2

# -- Grid
a = 2500.
L = 2.*np.pi * a
N = 24
grid = Grid(N, L)

# -- viscosity
nu = 0

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
axe = plotImageGF2( grid, k, r2, q2, f20=f20, nu=0, 
                    nIter=nIter, axe=plt.subplot(111)
                    )


