'''
Estimates correlation matrix with finite ensembles of perturbations and illustrates sampling noise inducing unphysical teleconnections.

Since the number of members is tightly constrained by integration cost in real atmospheric models, localization is often used to circumvent this problem by restricting the sampled covariance on a compact support.
'''
from sys import stdout
import numpy as np 
from numpy import pi
import matplotlib.pyplot as plt

from DM93 import modelSpPropagator
from DM93 import Uncorrelated, Foar, Soar, Gaussian

#====================================================================
#===| setup and configuration |======================================

execfile('config.py')

# -- forecast errors
fctLc = grid.L/20.
fctCorr = Gaussian(grid, fctLc)

# -- ensemble of perturbations
nList = [3, 10, 30, 100, 500]

#====================================================================
#===| computations |=================================================

perturbations = list()
BMatrices = dict()
vmin = np.infty
vmax = -np.infty
for n in nList:
    print('Ensemble of %d perturbations'%n)
    for i in xrange(n):
        if len(perturbations) <= i:
            stdout.write('..%d'%(i+1))
            stdout.flush()
            perturbations.append(fctCorr.random())

    cubeB = np.empty(shape=(n, grid.J, grid.J))
    for i, p in enumerate(perturbations):
        cubeB[i] = np.outer(p,p)

    B = np.mean(cubeB, axis=0)
    if vmin > B.min() : vmin = B.min()
    if vmax < B.max() : vmax = B.max()
    BMatrices[n] = B
    del cubeB
    print('\n')

#====================================================================
#===| plots |========================================================

fig = plt.figure(figsize=(8,10))
fig.subplots_adjust(wspace=0.01, hspace=0.35)
for i, n in enumerate(nList):
    axe = plt.subplot((len(nList)+1)/2, 2, i+1)
    axe.matshow(BMatrices[n], vmin=0, vmax=1)
    axe.set_xticks(())
    axe.set_yticks(())
    axe.set_title('$N=%d$'%n)
axe = plt.subplot((len(nList)+1)/2, 2, len(nList)+1)
axe.matshow(fctCorr.matrix, vmin=0, vmax=1)
axe.set_xticks(())
axe.set_yticks(())
axe.set_title('Exact correlation matrix')
fig.suptitle(fctCorr.name, fontsize=16)

plt.show()
