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
dt =6.*h

# -- zonal wind
U = 100.*km/h

# -- viscosity
nuFactor = 0.
nu =  nuFactor/dt*a**2

# -- initial state
ic = np.sin(2.*pi*grid.x/grid.L)

# -- integration 
MSpec = modelSpPropagator(grid, U, dt=dt, nu=nu)
M = np.dot(np.dot(grid.F, MSpec), grid.F.T)

nDt = 200

#====================================================================
#===| computations |=================================================
traj = np.empty(shape=(nDt+1, grid.J))
x = ic
for i in xrange(nDt+1):
    traj[i] = x
    x = np.dot(M, x)

#====================================================================
#===| plots |========================================================

fig = plt.figure()
axe = plt.subplot(111)
im = axe.matshow(traj, origin='lower')
axe.set_aspect('auto')
axe.set_xticks(())
axe.set_ylabel(r'$t$')
axe.set_ylabel(r'$x$')
plt.colorbar(im)

