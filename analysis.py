'''
Compute the analysis (through direct inversion of B+R innovation matrix) and output the error reduction.

For both observation and forecast errors, statistics need to be provided:

-   correlation model
-   correlation length
-   bias (0 by default)
-   variance (constant on the domain)

By default (and as it is a common hypothesis in most context), the observation error are uncorrelated.
What would be the impact of having correlated observation errors? The impact of biases?
'''
import numpy as np 
from numpy import pi
import matplotlib.pyplot as plt

from DM93 import Covariance, Uncorrelated, Foar, Soar, Gaussian

#====================================================================
#===| setup and configuration |======================================

execfile('config.py')

# -- observation errors
obsLc = None
obsCorr = Uncorrelated(grid, obsLc)
obsBias = 0.
obsVar = 1.

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

# -- reduction of error
error_b = grid.dx * np.sqrt(sum(fctErr**2))
error_a = grid.dx * np.sqrt(sum((xa-truth)**2))
print('background error = %.1e'%error_b)
print('analysis error = %.1e'%error_a)
print('error reduction = %.1f%%'%((error_b-error_a)/error_b*100.))

#====================================================================
#===| plots |========================================================

fig = plt.figure()
fig.subplots_adjust(wspace=0.05)
ax1 = plt.subplot(211)
ax2 = plt.subplot(212)

ax1.plot(grid.x, truth, color='k', linewidth=2, label='$x_t$')
ax1.plot(grid.x, xb, color='b', label='$x_b$')
ax1.plot(grid.x, y, color='g', marker='o', linestyle='none', label='$y$')
ax1.plot(grid.x, xa, color='r', linewidth=2, label='$x_a$')


ax2.plot(   grid.x, y-xb, color='m', marker='o', markersize=4, 
            linestyle='none', label='$y-x_b$')
ax2.plot(   grid.x, dxa, color='r', label='$\Delta x_a$')
ax2.plot(   grid.x, fctErr, color='b', linestyle=':', linewidth=3,
            label='$\epsilon_b$')
ax2.plot(   grid.x, xa-truth, color='r', linestyle=':', linewidth=3, 
            label='$\epsilon_a$')
ax2.axhline(y=0, color='k')

xticklabels, xticks = grid.ticks(units=km)[:2]
ax1.set_xticks(xticks)
ax1.set_xticklabels(())
ax2.set_xlabel('$x$ [km]')
ax2.set_xticks(xticks)
ax2.set_xticklabels(xticklabels)

ax1.legend(loc='best')
ax2.legend(loc='best')

plt.show()
