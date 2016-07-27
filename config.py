from numpy import pi 
from DM93 import Grid

# -- units of space: m and time: s
km = 1000.
h = 3600.
day = 24.*h

# -- discretization
L = 16000 * km
N = 48
grid = Grid(N, L)
dt =1.*h

# -- zonal wind
U = 20.*km/h

# -- viscosity
nuFactor = 0.
nu =  nuFactor/dt*(2.*pi*grid.L)**2
