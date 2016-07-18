import numpy as np 
from gridCls import Grid

def modelSpPropagator(grid, k=None, dt=1., nu=0):
    if k == None: k=grid.k
    return np.exp(2.*nu*np.pi*dt*k**2/grid.L**2)

def fcstSpVarPropagator(f2n, grid, r2, q2, k=None, dt=1., nu=0):
    m = modelSpPropagator(grid, k=k, dt=dt, nu=nu)
    return m**2*r2*f2n/(r2+f2n) + q2

#def spVarStationary(grid, r2, q2, k=None, dt=1., nu=0):
    
    
