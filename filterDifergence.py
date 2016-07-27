from sys import stdout
import numpy as np 
from numpy import pi
import matplotlib.pyplot as plt

from DM93 import Grid
from DM93 import AdvectionDiffusionModel
from DM93 import Covariance, Uncorrelated, Foar, Soar, Gaussian

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

# -- zonal wind
U = 100.*km/h

# -- viscosity
nuFactor = 0.0
nu =  nuFactor/dt*(2.*pi*grid.L)**2

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

# -- initial truth state
ampl = 10.
truIc = ampl * np.exp(-grid.x**2/(grid.L/6.)**2)

# -- model 
model = AdvectionDiffusionModel(grid, U, dt=dt, nu=nu)

# -- integration
nDt = 20

# -- XPs configurations
xpDict = {  'perf': {'modVar':0.0, 'doAss':True, 'label':'perfect model'},
            'perfNoDA': {'modVar':0.0, 'doAss':False, 'label':'perfect model without assimilation'},
            'imperf': {'modVar':0.01, 'doAss':True, 'label':'imperfect model'},
            }

fctVarTrajDict = dict()

#====================================================================
#===| computations |=================================================
for xpTag, xpConf in xpDict.iteritems():
    print(xpConf['label'])
    modVar = xpConf['modVar']
    doAssimilate = xpConf['doAss']

    # -- initialise seed
    np.random.seed(213134)
    
    # -- memory allocation
    times = np.array([i*dt for i in xrange(nDt+1)])
    
    truTraj = np.empty(shape=(nDt+1, grid.J))
    obsTraj = np.empty(shape=(nDt+1, grid.J))
    anlTraj = np.empty(shape=(nDt+1, grid.J))
    fctTraj = np.empty(shape=(nDt+1, grid.J))
    fctVarTraj = np.empty(nDt+1)
    
    
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
         
    fctVarTrajDict[xpTag] = fctVarTraj
    
    
    #================================================================
    #===| plots |====================================================
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
    
    varAx.plot(times/h, obsVar*np.ones(len(times)), linestyle='--', label=r'$\sigma_o^2$')
    varAx.plot(times/h, modVar*np.ones(len(times)), linestyle='--', label=r'$\sigma_q^2$')
    varAx.plot(times/h, fctVarTraj, label=r'$\sigma_b^2$')
    varAx.set_yscale('log')
    varAx.set_xlabel(r'$t$ [hours]')
    varAx.set_xticks(times[::nDt/nTimeTicks]/h)
    varAx.legend(loc='upper right')
    varAx.set_title('Forecast variance')
    
    title = (   xpConf['label'] + '\n' +
                r'$\sigma_q^2=%.0e,\ \sigma_b^2=%.0e,\ \sigma_b^2=%.0e$'%(
                                                    modVar, fctVar, obsVar))
    fig.suptitle(title, fontsize=16)
    fig.savefig('xpKF_%s.png'%xpTag)
    print('\n'+'='*30+'\n')

fig2 = plt.figure()
varAx2 = plt.subplot(111)
for xpTag, varTraj in fctVarTrajDict.iteritems():
    varAx2.plot(times/h, varTraj, label=xpDict[xpTag]['label'])
    
varAx2.plot(times/h, obsVar*np.ones(len(times)), linestyle='--', label=r'$\sigma_o^2$')

varAx2.legend(loc='upper right')
varAx2.set_yscale('log')
varAx2.set_xlabel(r'$t$ [hours]')
varAx2.set_xticks(times[::nDt/nTimeTicks]/h)
varAx2.set_title('Forecast variance')

fig2.savefig('xpKF_filterDiv.png')
plt.show()
