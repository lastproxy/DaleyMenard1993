import numpy as np 
from numpy import pi
import matplotlib.pyplot as plt

from DM93 import Grid
from DM93 import modelSpPropagator

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
dt =1.*h

# -- zonal wind
U = 20.*km/h

# -- viscosity
nuFactor = 0.
nu =  nuFactor/dt*a**2

# -- initial state
ic = np.sin(2.*pi*grid.x/grid.L)

# -- integration 
MSpec = modelSpPropagator(grid, U, dt=dt, nu=nu)
M = (grid.F.dot(MSpec)).dot(grid.F.T)

nDt = 200
times = np.array([i*dt for i in xrange(nDt+1)])

#====================================================================
#===| computations |=================================================
traj = np.empty(shape=(nDt+1, grid.J))
x = ic
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

nXTicks = 3
xticks = list()
xticklabels = list()
for i in xrange(nXTicks):
    gp = grid.J/(nXTicks-1)*i
    xticks.append(gp)
    if grid.x[gp] == 0:
        xticklabels.append('%.1e'%(grid.x[gp]/km))
    else:
        xticklabels.append('%.1e'%(grid.x[gp]/km))
axe.set_xticks(xticks)
axe.set_xticklabels(xticklabels)

plt.colorbar(im)
plt.show()

