'''
Defines constants and the periodic domain.

-   defines space and time units
-   defines the assimilation window for model integration
-   defines the periodic domaine and spectral truncature
-   define the two physical parameters: the zonal wind speed and the dissipation coefficient

`config.py` is sourced in most of the other scripts (such that they share the parametrisation), but one can replace the statetment `execfile('config.py')` with explicit local definitions of the grid and parameters.

The grid defined in the script is an instance of `Grid` class defined in `./DM93/gridCls.py`.
This object also provide the discrete Fourier transform and its inverse.
'''
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
U = 100.*km/h

# -- viscosity
#nuFactor = 0.00001
nuFactor = 0.0
nu =  nuFactor/dt*(2.*pi*grid.L)**2
