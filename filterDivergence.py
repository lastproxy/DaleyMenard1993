'''
Illustrates the forecast variance evolution and the Kalman Filter divergence issue by comparing three assimilation experiments:

1.  A perfect model assimilation initialised with an imperfect initial condition
2.  A perfect model integration initialised with an imperfect initial condition (no assimilation)
3.  An imperfect model assimilation initialised with an imperfect initial condition

As in `kalmanFilter.py`, statistics need to be provided.

By default, only the variance comparison plot is produced, change `doPlotXPs = True` for all three experiments to produce trajectory plots.
'''
from sys import stdout
import numpy as np 
from numpy import pi
import matplotlib.pyplot as plt

from DM93 import Grid
from DM93 import AdvectionDiffusionModel
from DM93 import Covariance, Uncorrelated, Foar, Soar, Gaussian

#====================================================================
#===| setup and configuration |======================================

execfile('config.py')

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
nDt = 10
times = np.array([i*dt for i in xrange(nDt+1)])

# -- XPs configurations
xpDict = {  'perf': {       'modVar':0.0, 'doAss':True, 
                            'label':'perfect model', 'color':'b'},
            'perfNoDA': {   'modVar':0.0, 'doAss':False, 
                            'label':'perfect model without assimilation', 'color':'g'},
            'imperf': {     'modVar':0.01, 'doAss':True, 
                            'label':'imperfect model', 'color':'r'},
            }

doPlotXPs = False
nTimeTicks = 5

#====================================================================
#===| computations |=================================================

fctVarTrajDict = dict()
anlVarTrajDict = dict()

for xpTag, xpConf in xpDict.iteritems():
    print(xpConf['label'])
    modVar = xpConf['modVar']
    doAssimilate = xpConf['doAss']

    # -- initialise seed
    np.random.seed(213134)
    
    # -- memory allocation
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
     

    fctVarTrajDict[xpTag] = fctVarTraj
    anlVarTrajDict[xpTag] = fctVarTraj
    
    
    #================================================================
    #===| plots |====================================================
    if doPlotXPs:
    
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
                    linestyle='--', color='g', label=r'$\sigma_o^2$')
        if modVar > 0:
            varAx.plot( times/h, modVar*np.ones(len(times)),
                        linestyle='--', color='m', label=r'$\sigma_q^2$')
        varAx.plot(times/h, fctVarTraj, color='b', label=r'$\sigma_f^2$')
        varAx.plot(times/h, anlVarTraj, color='r', label=r'$\sigma_a^2$')

        varAx.set_yscale('log')
        maxVar = max((obsVar, modVar, fctVarTraj.max(), anlVarTraj.max()))
        if modVar > 0:
            minVar = min((obsVar, modVar, fctVarTraj.min(), anlVarTraj.min()))
        else:
            minVar = min((obsVar, fctVarTraj.min(), anlVarTraj.min()))
        varAx.set_ylim(0.8*minVar, 1.2*maxVar)

        varAx.set_xlabel(r'$t$ [hours]')
        varAx.set_xticks(times[::nDt/nTimeTicks]/h)
        varAx.legend(loc='upper right')
        varAx.set_title('Forecast variance')
        
        if modVar > 0:
            title = (   xpConf['label'] + '\n' +
                        r'$\sigma_q^2=%.0e,\ \sigma_b^2=%.0e,\ \sigma_b^2=%.0e$'%(
                                                        modVar, fctVar, obsVar))
        else:
            title = (   xpConf['label'] + '\n' +
                        r'$\sigma_b^2=%.0e,\ \sigma_b^2=%.0e$'%(fctVar, obsVar))
        fig.suptitle(title, fontsize=16)
        fig.savefig('xpKF_%s.png'%xpTag)

    print('\n'+'='*30+'\n')





fig2 = plt.figure()
varAx2 = plt.subplot(111)
minVar = np.infty
maxVar = -np.infty
for xpTag  in fctVarTrajDict.iterkeys():
    modVar = xpDict[xpTag]['modVar']
    fctVarTraj = fctVarTrajDict[xpTag]
    anlVarTraj = anlVarTrajDict[xpTag]
    varAx2.plot(times/h, fctVarTraj, color=xpDict[xpTag]['color'], 
                label= xpDict[xpTag]['label'])
    varAx2.plot(times/h, anlVarTraj, color=xpDict[xpTag]['color']) 
    varAx2.plot(times/h, obsVar*np.ones(len(times)), linestyle='--', color=xpDict[xpTag]['color'])
    if modVar > 0:
        varAx2.plot(times/h, modVar*np.ones(len(times)), 
                    linestyle='-.', color=xpDict[xpTag]['color'])
    
        tmp = min((obsVar, modVar, fctVarTraj.min(), anlVarTraj.min()))
    else:
        tmp = min((obsVar, fctVarTraj.min(), anlVarTraj.min()))
    if tmp < minVar : minVar = tmp
    tmp = max((obsVar, modVar, fctVarTraj.max(), anlVarTraj.max()))
    if tmp > maxVar : maxVar = tmp

varAx2.set_yscale('log')
varAx2.set_ylim(0.8*minVar, 1.2*maxVar)

varAx2.legend(loc='upper right')
varAx2.set_xlabel(r'$t$ [hours]')
varAx2.set_xticks(times[::nDt/nTimeTicks]/h)
varAx2.set_title('Forecast variance')

fig2.savefig('xpKF_filterDiv.png')
plt.show()
