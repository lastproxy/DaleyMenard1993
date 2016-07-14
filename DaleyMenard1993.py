import numpy as np 
from gridCls import Grid
from corrModelLib import Uncorrelated, Soar

# -- Grid
a = 2500.
L = 2.*np.pi * a
N = 24
grid = Grid(N, L)

corrObs = Uncorrelated(grid)

lMod = a/6.
corrMod = Soar(grid, lMod)

plt.loglog(grid.k, corrMod.powSpec())
plt.loglog(grid.k, corrObs.powSpec())

