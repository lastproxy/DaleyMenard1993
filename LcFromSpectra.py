'''
Correlation length is not defined univocally.
One definition is based on the curvature of a gaussian at the origin.
Based on this definition, it is possible to determine the correlation length from the power spectrum of the correlation.

This script use this definition to estimate the correlation length from three correlation models and compare them with the true one.

What explains that one model is better than the other?
What happens when the spectral resolution is better?
'''
import numpy as np 
from numpy import pi, sqrt, arange
import matplotlib.pyplot as plt

from DM93 import Grid
from DM93 import Uncorrelated, Foar, Soar, Gaussian
from DM93 import specCorrLength


#====================================================================
#===| setup and configuration |======================================

# -- units of space: m 
km = 1000.
L = 16000 * km
N = 48
grid = Grid(N, L)


# -- correlation
Lc = grid.L/20.

#====================================================================
#===| computations |=================================================
print('Lc = %.2f km, N = %d'%(Lc/km, N))
print('='*10)

LcSpec = dict()
for CorrModel in (Foar, Soar, Gaussian):
    corr = CorrModel(grid, Lc)

    powSpec = corr.powSpecTh()
    LcSpec[corr.name] = specCorrLength(grid, powSpec)
    error = abs(LcSpec[corr.name]-Lc)/Lc*100.
    print('%s : Lc = %.2f km (%.2f%% error)'%(corr.name, LcSpec[corr.name]/km, error))
