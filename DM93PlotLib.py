import numpy as np
import matplotlib.pyplot as plt

from DM93Lib import (   fcstSpVarPropagator, spVarStationary, varItGenerator, 
                        analSpVar, convRateAssymp
                        )


def plotCorrPowSpectra(grid, r2, q2, axe=None, legend=True):
    ''' Plot error correlation power spectra
    
    :Parameters:
        grid : `Grid`
            Periodic 1D grid
        r2 : np.ndarray
            Observation error correlation power spectra 
        q2 : np.ndarray
            Model error correlation power spectra
    '''
    if axe==None:
        axe = plt.subplot(111)
    axe.semilogy(grid.k, r2, label='obs. error')
    axe.semilogy(grid.k, q2, label='model error')
    axe.set_title('Error correlation power spectra')
    axe.set_xlabel('wavenumber $k$')

    if legend : axe.legend(loc='upper right')
    return axe

    


def plotImageGF2(   grid, k, r2, q2, f20=0, dt=1., nu=0, nIter=5, 
                    axe=None, annotate=True, limAmpl=1.2, legend=True):
    ''' Plot G(f2) phase space and iterations convergence 
    
    :Parameters:
        grid : `Grid`
            Periodic 1D grid
        k : int
            Wavenumber
        r2 : np.ndarray
            Observation error correlation power spectra
        q2 : np.ndarray
            Model error correlation power spectra
        dt : float
            Time increment
        nu : float
            Viscosity coefficient
    '''
    if axe==None:
        axe = plt.subplot(111)

    convF2 = list()
    convG = list()
    
    f2n = f20
    for g in varItGenerator(grid, f20, r2[k], q2[k], nIter=nIter, 
                            k=k, dt=dt, nu=nu):
        convF2.append(f2n)
        f2n = g
        convG.append(f2n)

    # -- stationary solution
    f2Plus, f2Minus = spVarStationary(grid, r2[k], q2[k], k=k, dt=dt, nu=nu)
    GF2Plus = fcstSpVarPropagator(  grid, f2Plus, r2[k], q2[k], 
                                    k=k, dt=dt, nu=nu)
    GF2Minus = fcstSpVarPropagator( grid, f2Minus, r2[k], q2[k], 
                                    k=k, dt=dt, nu=nu)
    
    # -- plotting convergence iterates
    for i, (f2, g) in enumerate(zip(convF2, convG)):
        label = None
        #if i==nIter-1:  
        #    label = r'$G(f_n^2)$'
        axe.plot(f2, g, marker='o', color='b', label=label)
        if annotate: 
            axe.annotate(str(i), xy=(f2, g), fontsize=16, color='b')


    # -- plotting G(f2)
    minF2 = np.min((f2Plus, np.min(convF2)))
    maxF2 = np.max((f2Plus, np.max(convF2)))
    minG = np.min((GF2Plus, np.min(convG)))
    maxG = np.max((GF2Plus, np.max(convG)))
    domF2 = np.linspace(limAmpl*minF2, limAmpl*maxF2, 1000)
    imGF2 = list()
    for f2 in domF2:
        imGF2.append(fcstSpVarPropagator(   grid, f2, r2[k], q2[k],
                                            k=k, dt=dt, nu=nu))
    
    # -- stationary solutions
    axe.plot(f2Plus, GF2Plus, 's', color='g', label=r'$\overline{f}_+^2$')
    axe.axvline(x=f2Plus, linestyle='--', color='g', 
                )
    axe.axhline(y=GF2Plus, linestyle='--', color='g')

    axe.plot(f2Minus, GF2Minus, 's', color='r', label=r'$\overline{f}_-^2$')

    axe.plot(domF2, imGF2, 'k', linewidth=2, label=r'$G(f^2)$')
    axe.plot(domF2, domF2, 'k', linestyle=':')


    # -- finishing touches
    axe.set_xlabel(r'$f^2$')
    axe.set_ylabel(r'$G(f^2)$')
    axe.set_ylim((limAmpl*minG, limAmpl*maxG))
    
    axe.set_title(r'Convergence to stationary solution for $k=%d$'%k)

    if legend : axe.legend(loc='best')
    return axe


def plotAssympVar(  grid, r2, q2, dt=1., nu=0., axe=None, yscale='log', 
                    legend=True):
    ''' Plot assymptotical variances (forecast and analysis) spectra
    
    :Parameters:
        grid : `Grid`
            Periodic 1D grid
        r2 : np.ndarray
            Observation error correlation power spectra
        q2 : np.ndarray
            Model error correlation power spectra
        dt : float
            Time increment
        nu : float
            Viscosity coefficient
    '''
    if axe==None:
        axe = plt.subplot(111)

    f2Plus = spVarStationary(grid, r2, q2, dt=dt, nu=nu)[0]
    analPlus = analSpVar(f2Plus, r2, q2)

    axe.plot(grid.k, f2Plus, label=r'$\overline{f}_+^2$')
    axe.plot(grid.k, analPlus, label=r'$\overline{a}_+^2$')

    axe.set_yscale(yscale)
    axe.set_xlabel('wavenumber $k$')
    axe.set_title('Assymptotical variances')
    if legend: axe.legend(loc='upper right')

    return axe


def plotAssympConvRate( grid, r2, q2, dt=1., nu=0, axe=None, yscale='log', 
                        legend=True):
    ''' Plot assymptotical convergence in spectral space
    
    :Parameters:
        grid : `Grid`
            Periodic 1D grid
        r2 : np.ndarray
            Observation error correlation power spectra
        q2 : np.ndarray
            Model error correlation power spectra
        dt : float
            Time increment
        nu : float
            Viscosity coefficient
    '''
    
    if axe==None:
        axe = plt.subplot(111)
    
    cPlus = convRateAssymp(grid, r2, q2, dt=dt, nu=nu)
    axe.plot(grid.k, cPlus, label=r'$\overline{c}_+$')

    axe.set_yscale(yscale)
    axe.set_xlabel('wavenumber $k$')
    axe.set_title('Assymptotical convergence rate spectrum')
    if legend : axe.legend(loc='best')

    return axe

def plotViscAssympConv( grid, nuList, r2, q2, dt=1., axe=None, yscale='log',
                        legend=True):
    ''' Plot assymptotical convergence in spectral space for different 
    viscosity coefficients.
    
    :Parameters:
        grid : `Grid`
            Periodic 1D grid
        nuList : list(float)
            List of viscosity coefficients
        r2 : np.ndarray
            Observation error correlation power spectra
        q2 : np.ndarray
            Model error correlation power spectra
        dt : float
            Time increment
    '''
    
    if axe==None:
        axe = plt.subplot(111)
    
    for nu in nuList:
        cPlus = convRateAssymp(grid, r2, q2, dt=dt, nu=nu)
        axe.plot(grid.k, cPlus, label=r'$\nu=%.1e$'%nu)
    
    axe.set_yscale(yscale)
    axe.set_xlabel('wavenumber $k$')
    axe.set_title('Assymptotical convergence spectrum')
    if legend : axe.legend(loc='best')

    return axe


def plotViscAssympVar(  grid, nuList, r2, q2, dt=1., axe=None, yscale='log',
                        legend=True):   
    ''' Plot assymptotical forecast variance in spectral space for different 
    viscosity coefficients.
    
    :Parameters:
        grid : `Grid`
            Periodic 1D grid
        nuList : list(float) | np.ndarray(float)
            List of viscosity coefficients
        r2 : np.ndarray
            Observation error correlation power spectra
        q2 : np.ndarray
            Model error correlation power spectra
        dt : float
            Time increment
    '''
    
    if axe==None:
        axe = plt.subplot(111)
    
    for nu in nuList:
        f2Plus = spVarStationary(grid, r2, q2, dt=dt, nu=nu)[0]
        axe.plot(grid.k, f2Plus, label=r'$\nu=%.1e$'%nu)
    
    axe.set_yscale(yscale)
    axe.set_xlabel('wavenumber $k$')
    axe.set_title('Assymptotical forecast variance spectrum')
    if legend : axe.legend(loc='best')

    return axe
