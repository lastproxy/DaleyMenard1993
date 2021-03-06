'''
Run an assimilation cycle using the Kalman Filter; for each assimilation window: 

1.  observations are obtained (simulated from the truth plus a random realization of R);
2.  an analysis is computed;
3.  the analysis is integrated using the model to produce the forecast;
4.  covariance matrices are propagated using the Kalman Filter equations;
5.  the forecast and covariance is then used for the next assimilation window.

Observation, forecast and model errors statistics need to be provided:

-   correlation model
-   correlation length
-   bias (0 by default)
-   variance (constant on the domain)

The script plots the truth and forecast trajectories as well as the forecast and analysis variances evolution in time.
'''
from sys import stdout
import numpy as np 
from numpy import pi
import matplotlib.pyplot as plt

from DM93 import AdvectionDiffusionModel
from DM93 import Covariance, Uncorrelated, Foar, Soar, Gaussian

#====================================================================
#===| setup and configuration |======================================

execfile('config.py')

doAssimilate = True

# -- observation errors (R)
obsLc = None
obsCorr = Uncorrelated(grid)
obsBias = 0.
obsVar = 0.1

# -- forecast errors (B)
fctLc = grid.L/20.
fctCorr = Soar(grid, fctLc)
fctBias = 0.
fctVar = 2.

# -- model errors (Q)
modLc = grid.L/50.
modCorr = Gaussian(grid, modLc)
modBias = 0.
modVar = 0.01

# -- initial truth state
ampl = 10.
truIc = ampl * np.exp(-grid.x**2/(grid.L/6.)**2)

# -- model 
model = AdvectionDiffusionModel(grid, U, dt=dt, nu=nu)

# -- integration
nDt = 10

#====================================================================
#===| computations |=================================================

# -- memory allocation
times = np.array([i*dt for i in xrange(nDt+1)])

truTraj = np.empty(shape=(nDt+1, grid.J))
obsTraj = np.empty(shape=(nDt+1, grid.J))
anlTraj = np.empty(shape=(nDt+1, grid.J))
fctTraj = np.empty(shape=(nDt+1, grid.J))
fctVarTraj = np.empty(nDt+1)
anlVarTraj = np.empty(nDt+1)


# -- initial covariances
B = Covariance(grid, fctCorr.matrix * fctVar)
R = Covariance(grid, obsCorr.matrix * obsVar)
Q = Covariance(grid, modCorr.matrix * modVar)

# -- integration 
xt = truIc
xbIc = xt + B.random(bias=fctBias)
xb=xbIc
A = B
for i in xrange(nDt+1):
    stdout.write('..%d'%i)
    stdout.flush()

    # -- observations
    y = xt + R.random(bias=obsBias)

    if doAssimilate:

        # -- analysis using OI
        SInv = np.linalg.inv(B.matrix+R.matrix) 
        K = B.matrix.dot(SInv)
        xa = xb + K.dot(y-xb)
        
        # -- Kalman Filter
        B = Covariance(grid, model(A.matrix) + Q.matrix )
        A = Covariance(grid, (np.eye(grid.J) - K).dot(B.matrix)  )

    else:
        xa = xb

    # -- propagating
    xb = model(xa)
    xt = model(xt) + Q.random(bias=modBias) 
    
    # -- recording states
    truTraj[i] = xt
    obsTraj[i] = y
    anlTraj[i] = xa
    fctTraj[i] = xb
    fctVarTraj[i] = B.variance[0]
    anlVarTraj[i] = A.variance[0]
     


#====================================================================
#===| plots |========================================================
nTimeTicks = 5

fig = plt.figure(figsize=(8, 10))
fig.subplots_adjust(wspace=0.3, top=0.84)
truAx = plt.subplot(311)
fctAx = plt.subplot(312)
varAx = plt.subplot(313)

vmin = min((truTraj.min(), fctTraj.min()))
vmax = max((truTraj.max(), fctTraj.max()))

truAx.matshow(truTraj.T, origin='lower', vmin=vmin, vmax=vmax)
fctAx.matshow(fctTraj.T, origin='lower', vmin=vmin, vmax=vmax)

truAx.set_title('Truth')
fctAx.set_title('Forecasts')

gridTicksLabel, gridTicks, indexes = grid.ticks(units=km)
for axe in (truAx, fctAx):
    axe.set_aspect('auto')
    axe.xaxis.set_ticks_position('bottom')
    axe.set_xticks(())

    axe.set_ylabel(r'$x$ [km]')
    axe.set_yticks(indexes)
    axe.set_yticklabels(gridTicksLabel)


varAx.plot( times/h, obsVar*np.ones(len(times)), 
            linestyle='--', color='g',  label=r'$\sigma_o^2$')
if modVar > 0:
    varAx.plot( times/h, modVar*np.ones(len(times)), 
                linestyle='--', color='m', label=r'$\sigma_q^2$')
varAx.plot(times/h, fctVarTraj, color='b', label=r'$\sigma_f^2$')
varAx.plot(times/h, anlVarTraj, color='r', label=r'$\sigma_a^2$')

varAx.set_yscale('log')
varAx.set_xlabel(r'$t$ [hours]')

maxVar = max((obsVar, modVar, fctVarTraj.max(), anlVarTraj.max()))
if modVar > 0:
    minVar = min((obsVar, modVar, fctVarTraj.min(), anlVarTraj.min()))
else:
    minVar = min((obsVar, fctVarTraj.min(), anlVarTraj.min()))
    
varAx.set_ylim(0.8*minVar, 1.2*maxVar)
varAx.set_xticks(times[::nDt/nTimeTicks]/h)
varAx.legend(loc='upper right')
varAx.set_title('Forecast variance')

fig.suptitle(   r'$\sigma_q^2=%.0e,\ \sigma_b^2=%.0e,\ \sigma_o^2=%.0e$'%(
                                    modVar, fctVar, obsVar),
                fontsize=16
                )


plt.show()
