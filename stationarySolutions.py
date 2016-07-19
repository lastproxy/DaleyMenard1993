import numpy as np 
import matplotlib.pyplot as plt

from DM93 import Grid, Uncorrelated, Foar, Soar, Gaussian
from DM93 import spVarStationary, fcstSpVarPropagator,  varItGenerator

#====================================================================
#===| setup and configuration |======================================

# -- units of space: m and time: s
km = 1000.
h = 3600.

# -- discretization
a = 2500.*km
L = 2.*np.pi * a
N = 24
grid = Grid(N, L)
dt =6.*h

# -- viscosity
nuFactor = 0.
nu =  nuFactor/dt*a**2

# -- Correlations
corrObs = Uncorrelated(grid)

Lc = a/6.
corrMod = Soar(grid, Lc)

# -- correlation power spectra
r2 = corrObs.powSpec()
q2 = corrMod.powSpec()

# -- wavenumber and initial forecast variance
k = 10
f20 = 0.05

# -- number of iterations
nIter = 5

#====================================================================
#===| computations |=================================================

convF2 = list()
convG = list()

f2n = f20
for g in varItGenerator(grid, f20, r2[k], q2[k], nIter=nIter, 
                        k=k, dt=dt, nu=nu):
    convF2.append(f2n)
    f2n = g
    convG.append(f2n)


# -- stationary solutions
f2Plus, f2Minus = spVarStationary(grid, r2[k], q2[k], k=k, dt=dt, nu=nu)
GF2Plus = fcstSpVarPropagator(  grid, f2Plus, r2[k], q2[k], 
                                k=k, dt=dt, nu=nu)
GF2Minus = fcstSpVarPropagator( grid, f2Minus, r2[k], q2[k], 
                                k=k, dt=dt, nu=nu)



# -- variance domain of interest
minF2 = np.min((f2Plus, f2Minus, np.min(convF2)))
maxF2 = np.max((f2Plus, f2Minus, np.max(convF2)))
minG = np.min((GF2Plus, GF2Minus, np.min(convG)))
maxG = np.max((GF2Plus, GF2Minus, np.max(convG)))
domF2 = np.linspace(minF2, maxF2, 1000)

# -- image by G
imGF2 = list()
for f2 in domF2:
    imGF2.append(fcstSpVarPropagator(   grid, f2, r2[k], q2[k],
                                        k=k, dt=dt, nu=nu))

#====================================================================
#===| plots |========================================================

fig = plt.figure()
axe = plt.subplot(111)

# -- plotting G(f2)
axe.plot(domF2, imGF2, 'k', linewidth=2, label=r'$G(f^2)$')
axe.plot(domF2, domF2, 'k', linestyle=':')

# -- stationary solutions
axe.plot(f2Plus, GF2Plus, 's', color='g', label=r'$\overline{f}_+^2$')
axe.axvline(x=f2Plus, linestyle='--', color='g')
axe.axhline(y=GF2Plus, linestyle='--', color='g')

axe.plot(f2Minus, GF2Minus, 's', color='r', label=r'$\overline{f}_-^2$')

# -- plotting convergence iterates
for i, (f2, g) in enumerate(zip(convF2, convG)):
    label = None
    axe.plot(f2, g, marker='o', color='b', label=label)
    axe.annotate(str(i), xy=(f2, g), fontsize=16, color='b')

# -- finishing touches
axe.set_xlabel(r'$f^2$')
axe.set_ylabel(r'$G(f^2)$')

axe.set_title(r'Convergence to stationary solution for $k=%d$'%k)

axe.legend(loc='best')

plt.show()
