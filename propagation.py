from sys import stdout
import numpy as np 
from numpy import pi
import matplotlib.pyplot as plt

from DM93 import modelSpPropagator

#====================================================================
#===| setup and configuration |======================================

execfile('config.py')

# -- initial state
x0 = grid.L/5.
ic = np.exp(-(grid.x-x0)**2/(L/6.)**2)

# -- integration
nDt = 200

#====================================================================
#===| computations |=================================================

# -- spectral (Ms) and grid (M) propagators (M = F.Ms.F')
Ms = modelSpPropagator(grid, U, dt=dt, nu=nu)
M = (grid.F.dot(Ms)).dot(grid.F.T)

# -- integration
nTimeTicks = 5
dtTicks = nDt/nTimeTicks
times = np.array([i*dt for i in xrange(0,nDt+1,dtTicks)])

traj = np.empty(shape=(nDt+1, grid.J))
x = ic
# -- x_{n+1} = M.x_{n} 
for i in xrange(nDt+1):
    stdout.write('..%d'%i)
    stdout.flush()
    x = M.dot(x)
    traj[i] = x

#====================================================================
#===| plots |========================================================

fig = plt.figure()
axe = plt.subplot(111)
im = axe.matshow(traj, origin='lower')

axe.set_aspect('auto')

axe.set_ylabel(r'$t$ [hours]')
#axe.set_yticklabels(times/h)
axe.set_yticks(times/h)

axe.set_xlabel(r'$x$ [km]')
axe.xaxis.set_ticks_position('bottom')

xticklabels, xticks, indexes = grid.ticks(units=km)
axe.set_xticks(indexes)
axe.set_xticklabels(xticklabels)

plt.colorbar(im)
plt.show()

