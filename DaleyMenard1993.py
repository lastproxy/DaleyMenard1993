import numpy as np 
from numpy import pi, cos, sin, sqrt
from numpy.fft import fft, ifft 


#===| Defining spaces |==============================================
def buildGrid(L, N):
    return np.array([L*j/(2.*N +1.) for j in xrange(-N, N+1)])

def buildSpectral(L, N):
    return np.array(range(-N, N+1), dtype=float)

def buildFourierTransforms(N, grid, L):
    J = 2*N+1
    assert len(grid) == J

    F = np.empty(shape=(J,J))
    for j in xrange(-N, N+1):
        F[j,0] = 1./sqrt(2.)
        for n in xrange(1,N+1):
            F[j,2*n-1]  = cos(2.*pi*n*grid[j]/L)
            F[j,2*n]    = sin(2.*pi*n*grid[j]/L)
    F *= sqrt(2./(2.*N+1.))
    return F
                

#===| Variance propagators |=========================================

# -- spectral model variance propagator (stationary)
def modelVarProp(p, dt=1., nu=1, L=1.):
    return np.exp(-2.*nu*np.pi**2*dt*p**2/L**2)

# -- forecast variance propagator [EQ 2.11]
def fcstVarProp(f2n, r2, q2, dt=1., nu=1, L=1.):
    m2p = modelVarProp(p, dt=dt, nu=nu, L=L)
    return m2p*r2*f2n / (r2 + f2n) + q2



#===| Correlation models |===========================================

# - uncorrelated
def uncorr(r):
    x = np.zeros(len(r))
    N = (len(x)-1)/2
    x[N] = 1.
    return x

def uncorrSpec(k):
    sp = np.ones(len(k))/len(k)
    return sp

# -- second order autoregressive
def soar(r, l):
    x = np.abs(r)/l
    return (1.+ x)*np.exp(-x)

def soarSpec(k, l, L):
    q = 2.*pi*k/L
    sp = (1. + q**2*l**2)**-2
    sp /= sum(sp)
    return sp


#===| MAIN |=========================================================
if __name__ == "__main__":

    # -- Grid
    a = 2500.
    L = 2.*np.pi * a
    N = 24
    J = 2*N+1
    x = buildGrid(L, N)

    # -- spectral space
    k = buildSpectral(L, N)
    F = buildFourierTransforms(N, x, L)
    FInv = F.transpose()

    # -- assert F unitary
    np.testing.assert_array_almost_equal(np.linalg.inv(F), FInv, decimal=14)

    # -- physical parameters
    nu = 0.


    # -- correlation model
    lObs = 0. 
    corrObs = np.zeros(J)
    rSpec = uncorrSpec(k)

    lMod = a/6.
    corrMod = soar(x, lMod)
    qSpec = soarSpec(k, lMod, L)
    
    plt.loglog(range(N+1), qSpec[N:])
    plt.loglog(range(N+1), rSpec[N:])
    
