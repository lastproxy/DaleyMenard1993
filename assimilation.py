from sys import stdout
import numpy as np 
from numpy import pi
import matplotlib.pyplot as plt

from DM93 import modelGridPropagator
from DM93 import Covariance, Uncorrelated, Foar, Soar, Gaussian

#====================================================================
#===| setup and configuration |======================================

execfile('config.py')

doAssimilate = False
#doAssimilate = True

# -- observation errors (R)
obsLc = None
obsCorr = Uncorrelated(grid)
obsErrBias = 0.
obsErrVar = 0.1

# -- forecast errors (B)
fctLc = grid.L/20.
fctCorr = Soar(grid, fctLc)
fctErrBias = 0.
fctErrVar = 2.

# -- model errors (Q)
modLc = grid.L/50.
modCorr = Gaussian(grid, modLc)
modErrBias = 0.
modErrVar = 0.5

# -- initial truth state
ampl = 10.
truIc = ampl * np.exp(-grid.x**2/(grid.L/6.)**2)

# -- integration
nDt = 20

#====================================================================
#===| computations |=================================================

# -- memory allocation
nTimeTicks = 5
dtTicks = nDt/nTimeTicks
times = np.array([i*dt for i in xrange(0,nDt+1,dtTicks)])

truTraj = np.empty(shape=(nDt+1, grid.J))
obsTraj = np.empty(shape=(nDt+1, grid.J))
anlTraj = np.empty(shape=(nDt+1, grid.J))
fctTraj = np.empty(shape=(nDt+1, grid.J))

# -- initial covariances
B = Covariance(grid, fctCorr.matrix * fctErrVar)
R = Covariance(grid, obsCorr.matrix * obsErrVar)
Q = Covariance(grid, modCorr.matrix * modErrVar)

# -- model grid propagator
M = modelGridPropagator(grid, U, dt=dt, nu=nu)

# -- integration 
xt = truIc
xbIc = xt + B.random()
xb=xbIc
for i in xrange(nDt+1):
    stdout.write('..%d'%i)
    stdout.flush()

    # -- observations
    y = xt + R.random()

    if doAssimilate:

        # -- analysis using OI
        SInv = np.linalg.inv(B.matrix+R.matrix) 
        K = B.matrix.dot(SInv)
        xa = xb + K.dot(y-xb)
        
        # -- Kalman Filter
        B = Covariance(grid, (M.dot(B.matrix)).dot(M.T) + Q.matrix )
        A = Covariance(grid, (np.eye(grid.J) - K).dot(B.matrix)  )

    else:
        xa = xb

    # -- propagating
    xb = M.dot(xa) + Q.random()
    xt = M.dot(xt) 
    
    # -- recording states
    truTraj[i] = xt
    obsTraj[i] = y
    anlTraj[i] = xa
    fctTraj[i] = xb
     


#====================================================================
#===| plots |========================================================

fig = plt.figure()
fig.subplots_adjust(wspace=0.3)
truAx = plt.subplot(121)
fctAx = plt.subplot(122)

vmin = min((truTraj.min(), fctTraj.min()))
vmax = max((truTraj.max(), fctTraj.max()))

truAx.matshow(truTraj, origin='lower', vmin=vmin, vmax=vmax)
fctAx.matshow(fctTraj, origin='lower', vmin=vmin, vmax=vmax)

truAx.set_title('Truth')
fctAx.set_title('Forecasts')

xticklabels, xticks, indexes = grid.ticks(units=km)
for axe in (truAx, fctAx):
    axe.set_aspect('auto')
    axe.set_ylabel(r'$t$ [hours]')
    #axe.set_yticklabels(times/h)
    axe.set_yticks(times/h)
    axe.set_xlabel(r'$x$ [km]')
    axe.xaxis.set_ticks_position('bottom')
    axe.set_xticks(indexes)
    axe.set_xticklabels(xticklabels)


