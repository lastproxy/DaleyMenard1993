import numpy as np 
from numpy import pi
import matplotlib.pyplot as plt

from DM93 import modelSpPropagator
from DM93 import Uncorrelated, Foar, Soar, Gaussian

#====================================================================
#===| setup and configuration |======================================

execfile('config.py')

# -- model errors
modLc = grid.L/20.
modCorr = Soar(grid, modLc)

# -- ensemble of perturbations
nList = [3, 10, 30, 100, 500]

#====================================================================
#===| computations |=================================================

perturbations = list()
BMatrices = dict()
for n in nList:
    for i in xrange(n):
        if len(perturbations) < i:
            perturbations.append(modCorr.random())

    cubeB = np.empty(shape=(n, grid.J, grid.J))
    for i, p in enumerate(perturbations):
        cubeB[i] = np.outer(p,p)

    BMatrices[n] = np.mean(cubeB, axis=0)
    del cubeB

#====================================================================
#===| plots |========================================================

fig = plt.figure()
fig.subplots_adjust(wspace=0.01, hspace=0.35)
for i, n in enumerate(nList):
    axe = plt.subplot((len(nList)+1)/2, 2, i+1)
    axe.matshow(BMatrices[n], vmin=0, vmax=1)
    axe.set_xticks(())
    axe.set_yticks(())
    axe.set_title('$N=%d$'%n)
axe = plt.subplot((len(nList)+1)/2, 2, len(nList)+1)
axe.matshow(modCorr.matrix, vmin=0, vmax=1)
axe.set_xticks(())
axe.set_yticks(())
axe.set_title('Exact correlation matrix')

plt.show()
