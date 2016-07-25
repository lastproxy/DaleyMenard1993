import numpy as np 
from gridCls import Grid


def analSpVar(f2n, r2, q2):
    ''' Analysis spectral variance 
    
    :Parameters:
        f2n : float | np.ndarray
            Initial forecast variance.
            If float provided, both `r2` and `q2` must be `float` as well, 
            the corresponding wavenumber power spectrum component.
        r2 : float | np.ndarray
            Observation error correlation power spectra or component
        q2 : float | np.ndarray
            Model error correlation power spectra or component
    '''
    return (r2*f2n)/(r2+f2n)

def modelSpPropagator(grid, k=None, dt=1., nu=0):
    ''' Model spectral propagator

    :Parameters:
        grid : `Grid`
            Periodic 1D grid
        k : int | None
            If not provided (or == None), then all spectrum is propagated.
        dt : float
            Time increment
        nu : float
            Viscosity coefficient
    '''
    if k==None:
        k = grid.halfK
    return np.exp(-2.*nu*np.pi*dt*k**2/grid.L**2)

def fcstSpVarPropagator(grid, f2n, r2, q2, k=None, dt=1., nu=0):
    ''' Forecast variance propagator 
   
    (noted G in the article)
    
    :Parameters:
        grid : `Grid`
            Periodic 1D grid
        f2n : float | np.ndarray
            Initial forecast variance.
            If float provided, both `r2` and `q2` must be `float` as well, 
            the corresponding wavenumber power spectrum component, and `k`
            must be provided as an `int` (the wavenumber).
        r2 : float | np.ndarray
            Observation error correlation power spectra or component
        q2 : float | np.ndarray
            Model error correlation power spectra or component
        k : int | None
            If not provided (or == None), then all spectrum is propagated
            and `f2n`, `r2` and `q2` must be arrays (full spectra).
        dt : float
            Time increment
        nu : float
            Viscosity coefficient
    '''
    if isinstance(f2n, (float, int)) or isinstance(k, int):
        assert isinstance(r2, (float, int))
        assert isinstance(q2, (float, int))
        assert isinstance(k, int)
        assert isinstance(f2n, (float, int))
    else:
        assert isinstance(r2, np.ndarray)
        assert isinstance(q2, np.ndarray)
        assert k == None
        assert isinstance(f2n, np.ndarray)
        

    m = modelSpPropagator(grid, k=k, dt=dt, nu=nu)
    return m**2*r2*f2n/(r2+f2n) + q2



def varItGenerator(grid, f2n, r2, q2, nIter=10, k=None, dt=1., nu=0):
    ''' Forecast variance iteration generator
    
    :Parameters:
        grid : `Grid`
            Periodic 1D grid
        f2n : float | np.ndarray
            Initial forecast variance.
            If float provided, both `r2` and `q2` must be `float` as well, 
            the corresponding wavenumber power spectrum component, and `k`
            must be provided as an `int` (the wavenumber).
        r2 : float | np.ndarray
            Observation error correlation power spectra or component
        q2 : float | np.ndarray
            Model error correlation power spectra or component
        nIter : int
            Maximal number of iterations
        k : int | None
            If not provided (or == None), then all spectrum is propagated
            and `f2n`, `r2` and `q2` must be arrays (full spectra).
        dt : float
            Time increment
        nu : float
            Viscosity coefficient
    '''
    i = 0
    while i < nIter:
        f2n = fcstSpVarPropagator(grid, f2n, r2, q2, k=k, dt=dt, nu=nu)
        yield f2n
        i += 1


def spVarStationary(grid, r2, q2, k=None, dt=1., nu=0):
    ''' Spectral variance stationary solutions.
    Returns the two solutions, the first being the physical stable one, 
    the second unphysical and unstable.

    :Parameters:
        grid : `Grid`
            Periodic 1D grid
        r2 : float | np.ndarray
            Observation error correlation power spectra or component
            If float provided, `q2` must be `float` as well, 
            the corresponding wavenumber power spectrum component, and `k`
            must be provided as an `int` (the wavenumber).
        q2 : float | np.ndarray
            Model error correlation power spectra or component
        k : int | None
            If not provided (or == None), then all spectrum is propagated
            and `f2n`, `r2` and `q2` must be arrays (full spectra).
        dt : float
            Time increment
        nu : float
            Viscosity coefficient
    '''
    m = modelSpPropagator(grid, k=k, dt=dt, nu=nu)
    alpha = 0.5 * (q2 + r2*(m**2+1.))
    beta = alpha**2 - m**2*r2**2
    return (    alpha - r2 + np.sqrt(beta),
                alpha - r2 - np.sqrt(beta)
                )
    
def convRate(grid, f2n, r2, q2, k=None, dt=1., nu=0):
    ''' Convergence rate
    
    :Parameters:
        f2n : float | np.ndarray
            Initial forecast variance.
            If float provided, `q2` must be `float` as well, 
            the corresponding wavenumber power spectrum component, and `k`
            must be provided as an `int` (the wavenumber).
        r2 : float | np.ndarray
            Observation error correlation power spectra or component
        q2 : float | np.ndarray
            Model error correlation power spectra or component
        k : int | None
            If not provided (or == None), then all spectrum is propagated
            and `f2n`, `r2` and `q2` must be arrays (full spectra).
        dt : float
            Time increment
        nu : float
            Viscosity coefficient
    '''
    m = modelSpPropagator(grid, k=k, dt=dt, nu=nu)
    return (m**2*r2 + q2 - f2n)/(f2n + r2)


def convRateAssymp(grid, r2, q2, k=None, dt=1., nu=0):
    ''' Assymptotic convergence rate
    
    :Parameters:
        f2n : float | np.ndarray
            Initial forecast variance.
            If float provided, `q2` must be `float` as well, 
            the corresponding wavenumber power spectrum component, and `k`
            must be provided as an `int` (the wavenumber).
        r2 : float | np.ndarray
            Observation error correlation power spectra or component
        q2 : float | np.ndarray
            Model error correlation power spectra or component
        k : int | None
            If not provided (or == None), then all spectrum is propagated
            and `f2n`, `r2` and `q2` must be arrays (full spectra).
        dt : float
            Time increment
        nu : float
            Viscosity coefficient
    '''
    m = modelSpPropagator(grid, k=k, dt=dt, nu=nu)
    alpha = 0.5 * (q2 + r2*(m**2+1.))
    beta = alpha**2 - m**2*r2**2
    return (alpha - np.sqrt(beta))/(alpha + np.sqrt(beta))

