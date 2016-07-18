import numpy as np 
from gridCls import Grid
from corrModelLib import Uncorrelated, Soar
from DM93Lib import fcstSpVarPropagator as G


# -- Grid
a = 2500.
L = 2.*np.pi * a
N = 24
grid = Grid(N, L)

# -- Correlations
corrObs = Uncorrelated(grid)

lMod = a/6.
corrMod = Soar(grid, lMod)


r2 = corrObs.powSpec()
q2 = corrMod.powSpec()

#plt.figure()
#plt.loglog(grid.k, q2)
#plt.loglog(grid.k, r2)


# -- convergence
f20 = zeros(grid.N+1)



convF2 = list()
convG = list()

f2n = f20
for i in xrange(10):
    convF2.append(f2n)
    f2n = G(f2n, grid, r2, q2)
    convG.append(f2n)

plt.figure()
for f2, g in zip(convF2, convG):
    plt.plot(f2[0], g[0], 'ob')
    plt.plot(f2[N], g[N], 'or')
