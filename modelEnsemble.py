from sys import stdout
import numpy as np 
from numpy import pi
import matplotlib.pyplot as plt

from DM93 import Grid
from DM93 import AdvectionDiffusionModel

#====================================================================
#===| setup and configuration |======================================

# -- units of space: m and time: s
km = 1000.
h = 3600.
day = 24.*h

# -- discretization
L = 16000 * km
N = 48
grid = Grid(N, L)
dt =1.*h

# -- initial state
x0 = grid.L/5.
ic = np.exp(-(grid.x-x0)**2/(L/6.)**2)


# -- ensemble of models
nModels = 20
models = list()
for i in xrange(nModels):
    nuFactor = np.random.normal(0.00001, scale=0.000001)
    nu =  nuFactor/dt*(2.*pi*grid.L)**2
    U = np.random.normal(100., scale=10.)*km/h
    models.append(AdvectionDiffusionModel(grid, U, dt=dt, nu=nu))




#====================================================================
#===| computations |=================================================

# -- 1 dt integration

finalStates = np.empty(shape=(nModels, grid.J))
for n in xrange(nModels):
    finalStates[n] = models[n](ic)

meanState =  finalStates.mean(axis=0)
deviations =  np.empty(shape=(nModels, grid.J))
for n in xrange(nModels):
    deviations[n] = finalStates[n] - meanState


#====================================================================
#===| plots |========================================================

for n in xrange(nModels):
    plt.plot(grid.x, deviations[n])
