import numpy as np
import matplotlib.pyplot as plt

from DM93Lib import fcstSpVarPropagator as G
from DM93Lib import modelSpPropagator as M
from DM93Lib import spVarStationary, varItGenerator 


def plotCorrPowSpectra(grid, r2, q2, axe=None):
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
    axe.legend(loc='upper right')
    axe.set_title('Error correlation power spectra')
    axe.set_xlabel('wavenumber $k$')
    return axe

    


def plotImageGF2(   grid, k, r2, q2, f20=0, nu=0, nIter=5, 
                    axe=None, annotate=True):
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
        nu : float
            Viscosity coefficient
    '''
    if axe==None:
        axe = plt.subplot(111)

    convF2 = list()
    convG = list()
    
    f2n = f20
    for g in varItGenerator(grid, f20, r2[k], q2[k], nIter=nIter, k=k, nu=nu):
        convF2.append(f2n)
        f2n = g
        convG.append(f2n)

    # -- stationary solution
    f2Plus, f2Minus = spVarStationary(grid, r2[k], q2[k], k=k, nu=nu)
    GF2Plus = G(grid, f2Plus, r2[k], q2[k], k=k, nu=nu)
    GF2Minus = G(grid, f2Minus, r2[k], q2[k], k=k, nu=nu)
    
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
    domF2 = np.linspace(1.1*minF2, 1.1*maxF2, 1000)
    imGF2 = list()
    for f2 in domF2:
        imGF2.append(G(grid, f2, r2[k], q2[k], k=k, nu=nu))
    
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
    axe.set_ylim((1.1*minG, 1.1*maxG))
    
    axe.set_title(r'Convergence to stationary solution for $k=%d$'%k)

    plt.legend(loc='best')
    return axe
