#-------------------------- LICENCE BEGIN ---------------------------
# This file is part of DaleyMenard93.
#
# DaleyMenard93 is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DaleyMenard93 is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with DaleyMenard93.  If not, see <http://www.gnu.org/licenses/>.
#
# Authors - Martin Deshaies-Jacques, Richard Menard
#
# Copyright 2016 - Air Quality Research Division, Environnement Canada
#-------------------------- LICENCE END -----------------------------
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

def modelSpPropagator(grid, U, dt=1., nu=0):
    ''' Model spectral propagator

    :Parameters:
        grid : `Grid`
            Periodic 1D grid
        U : float
            Constant zonal wind speed
        dt : float
            Time increment
        nu : float
            Viscosity coefficient
    '''
    MSpec = np.zeros(shape=(grid.J, grid.J))
    MSpec[0,0] = 1.
    for j in xrange(1,grid.N+1):
        phi = 2.*np.pi*j*U*dt/grid.L
        ampl = np.exp(-4.*np.pi**2*nu*dt*j**2/grid.L**2)
        MSpec[2*j-1, 2*j-1] = ampl * np.cos(phi)
        MSpec[2*j, 2*j-1]   = ampl * np.sin(phi)
        MSpec[2*j-1, 2*j]   = -ampl * np.sin(phi)
        MSpec[2*j, 2*j]     = ampl * np.cos(phi)
    return MSpec

def modelGridPropagator(grid, U, dt=1., nu=0):
    ''' Model grid-space propagator

    The grid-space propagator is obtained from the spectral propagator
    by inverse 2D Fourier transform.  
    If S is the spectral propagator and F the Fourier transform matrix
    and F' its transpose (and inverse), then M the grid-space propagator
    is simply: M = F.S.F'

    :Parameters:
        grid : `Grid`
            Periodic 1D grid
        U : float
            Constant zonal wind speed
        dt : float
            Time increment
        nu : float
            Viscosity coefficient
    '''
    MSpec = modelSpPropagator(grid, U, dt=dt, nu=nu)
    M = np.dot(np.dot(grid.F, MSpec), grid.F.T)
    return M

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
       
    if k==None: k = grid.halfK
    m2 = np.exp(-4.*nu*np.pi*dt*k**2/grid.L**2)
    return m2*r2*f2n/(r2+f2n) + q2



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
    if k==None: k = grid.halfK
    m2 = np.exp(-4.*nu*np.pi*dt*k**2/grid.L**2)
    alpha = 0.5 * (q2 + r2*(m2+1.))
    beta = alpha**2 - m2*r2**2
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
    if k==None: k = grid.halfK
    m2 = np.exp(-4.*nu*np.pi*dt*k**2/grid.L**2)
    return (m2*r2 + q2 - f2n)/(f2n + r2)


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
    if k==None: k = grid.halfK
    m2 = np.exp(-4.*nu*np.pi*dt*k**2/grid.L**2)
    alpha = 0.5 * (q2 + r2*(m2+1.))
    beta = alpha**2 - m2*r2**2
    return (alpha - np.sqrt(beta))/(alpha + np.sqrt(beta))

