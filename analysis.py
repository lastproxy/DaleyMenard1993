import numpy as np 
from numpy import pi
import matplotlib.pyplot as plt

from DM93 import Covariance, Uncorrelated, Foar, Soar, Gaussian

#====================================================================
#===| setup and configuration |======================================

execfile('config.py')

# -- observation errors
obsCorr = Uncorrelated(grid)
obsBias = 0.
obsVar = 0.1

# -- forecast errors
fctLc = grid.L/20.
fctCorr = Soar(grid, fctLc)
fctBias = 0.
fctVar = 2.

# -- initial truth state
ampl = 10.
truth = ampl * np.exp(-grid.x**2/(grid.L/6.)**2)

#====================================================================
#===| computations |=================================================

# -- covariance matrices
B = Covariance(grid, fctVar * fctCorr.matrix)
R = Covariance(grid, obsVar * obsCorr.matrix)

# -- random error structures
fctErr = B.random(bias=fctBias)
obsErr = R.random(bias=obsBias)

# -- background state
xb = truth + fctErr

# -- observations
y = truth + obsErr

# -- analysis
SInv = np.linalg.inv(B.matrix+R.matrix) 
K = B.matrix.dot(SInv)
dxa = K.dot(y-xb) 

xa = xb + dxa

#====================================================================
#===| plots |========================================================

fig = plt.figure()
axe = plt.subplot(111)

axe.plot(grid.x, truth, color='k', linewidth=2, label='$x_t$')
axe.plot(grid.x, xb, color='b', label='$x_b$')
axe.plot(grid.x, y, color='g', marker='o', linestyle='none', label='$y$')
axe.plot(grid.x, xa, color='r', linewidth=2, label='$x_a$')

xticklabels, xticks = grid.ticks(units=km)[:2]
axe.set_xlabel('$x$ [km]')
axe.set_xticks(xticks)
axe.set_xticklabels(xticklabels)
axe.legend(loc='best')

plt.show()
