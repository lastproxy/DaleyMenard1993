from sys import stdout
import numpy as np 
from numpy import pi
import matplotlib.pyplot as plt

from DM93 import AdvectionDiffusionModel

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
model = AdvectionDiffusionModel(grid, U, dt=dt, nu=nu)

# -- integration
times = np.array([i*dt for i in xrange(nDt+1)])

traj = np.empty(shape=(nDt+1, grid.J))
x = ic
# -- x_{n+1} = M.x_{n} 
for i in xrange(nDt+1):
    stdout.write('..%d'%i)
    stdout.flush()
    x = model(x)
    traj[i] = x

#====================================================================
#===| plots |========================================================
nTimeTicks = 5

fig = plt.figure(figsize=(8,5))
axe = plt.subplot(111)
im = axe.matshow(traj.T, origin='lower')

axe.set_aspect('auto')


axe.set_xlabel(r'$t$ [hours]')
axe.set_xticks(times[::nDt/nTimeTicks]/h)
axe.xaxis.set_ticks_position('bottom')

gridTicksLabel, girdTicks, indexes = grid.ticks(units=km)
axe.set_ylabel(r'$x$ [km]')
axe.set_yticks(indexes)
axe.set_yticklabels(gridTicksLabel)



plt.colorbar(im)
plt.show()

