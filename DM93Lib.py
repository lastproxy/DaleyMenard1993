import numpy as np 
from gridCls import Grid

def modelSpPropagator(grid, dt=1., nu=0):
    return np.exp(2.*nu*np.pi*dt*grid.k**2/grid.L**2)

def fcstSpVarPropagator(f2n, grid, r2, q2, dt=1., nu=0):
    m = modelSpPropagator(grid, dt=dt, nu=nu)
    return m**2*r2*f2n/(r2+f2n) + q2


