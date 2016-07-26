import numpy as np 
from numpy import pi
import matplotlib.pyplot as plt

from DM93 import modelSpPropagator

#====================================================================
#===| setup and configuration |======================================

execfile('config.py')

# -- initial state
#ic = np.sin(2.*pi*grid.x/grid.L)
ic = np.exp(-grid.x**2/(L/6.)**2)

# -- integration
nDt = 200

#====================================================================
#===| computations |=================================================

# -- spectral (Ms) and grid (M) propagators (M = F.Ms.F')
Ms = modelSpPropagator(grid, U, dt=dt, nu=nu)
M = (grid.F.dot(Ms)).dot(grid.F.T)

# -- integration
times = np.array([i*dt for i in xrange(nDt+1)])

traj = np.empty(shape=(nDt+1, grid.J))
x = ic
# -- x_{n+1} = M.x_{n} 
for i in xrange(nDt+1):
    traj[i] = x
    x = M.dot(x)

#====================================================================
#===| plots |========================================================

fig = plt.figure()
axe = plt.subplot(111)
im = axe.matshow(traj, origin='lower')

axe.set_aspect('auto')

axe.set_ylabel(r'$t$ [hours]')
axe.set_yticklabels(times/h)

axe.set_xlabel(r'$x$ [km]')
axe.xaxis.set_ticks_position('bottom')

xticklabels, xticks, indexes = grid.ticks(3, units=km)
axe.set_xticks(indexes)
axe.set_xticklabels(xticklabels)

plt.colorbar(im)
plt.show()

