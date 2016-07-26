import numpy as np 
from numpy import pi
import matplotlib.pyplot as plt

from DM93 import modelSpPropagator
from DM93 import Uncorrelated, Foar, Soar, Gaussian

#====================================================================
#===| setup and configuration |======================================

execfile('config.py')

# -- observation errors
obsCorr = Uncorrelated(grid)
obsErrBias = 0.
obsErrVar = 0.1

# -- model errors
modLc = grid.L/20.
modCorr = Soar(grid, modLc)
modErrBias = 0.
modErrVar = 2.

# -- initial truth state
ampl = 10.
truth = ampl * np.exp(-grid.x**2/(grid.L/6.)**2)

#====================================================================
#===| computations |=================================================

# -- random error structures
obsErr = obsCorr.random(variance=obsErrVar, mean=obsErrBias)
modErr = modCorr.random(variance=modErrVar, mean=modErrBias)

# -- covariance matrices
B = modErrVar * modCorr.matrix
R = obsErrVar * obsCorr.matrix

# -- background state
xb = truth + modErr

# -- observations
y = truth + obsErr

# -- analysis
SInv = np.linalg.inv(B+R) 
K = B.dot(SInv)
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
