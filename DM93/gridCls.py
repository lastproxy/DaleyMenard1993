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

class Grid(object):
    ''' Simple centered periodic grid class

    :Attributes:
        N : int
            spectral resolution (truncature)
        L : float 
            domain length (period)
        J : int
            number of grid points
        k : numpy.ndarray(int)
            wavenumber space
        x : numpy.ndarray(float)
            grid space
        dx : float
            grid space increment
        F : numpy.ndarray(float)
            Fourier transform matrix (unitary)

    :Methods:
        transform : numpy.ndarray(shape=self.J)
            return direct Fourier transform of signal.
            The decomposition order is such that the first element is 
            the constant term followed by sine and cosine terms. 
        inverse : numpy.ndarray(shape=self.J)
            return inverse Fourier transform of spectra
        ticks : int, format
            return a tuple (xticks, xticklabels) for axe formating
            
    '''
    
    def __init__(self, N, L):
        self.N=N
        self.L=L
        self.J = 2*self.N+1 


        self.halfK = np.array(range(self.N+1), dtype=float)
        self.k = np.array(range(-self.N,self.N+1), dtype=float)

        self.x = np.array(  [   self.L*k/(2.*self.N +1.) 
                                for k in self.k
                                ])
        self.dx = self.x[1]-self.x[0]
        
        self.F = self._fourierMatrix()

    def _fourierMatrix(self):
        ''' Build real unitary Fourier matrix '''
        F = np.zeros(shape=(self.J, self.J))
        
        F[:,0] = 1./np.sqrt(2.)
        for j in xrange(self.N+1):
            for n in xrange(1, self.N+1):
                F[j, 2*n-1]         = np.cos(2.*np.pi*n*self.x[j]/(self.L))
                F[j, 2*n]           = np.sin(2.*np.pi*n*self.x[j]/(self.L))
                F[self.N+j, 2*n-1]  = np.cos(2.*np.pi*n*self.x[self.N+j]/(self.L))
                F[self.N+j, 2*n]    = np.sin(2.*np.pi*n*self.x[self.N+j]/(self.L))

        F *= np.sqrt(2./self.J)

        # -- test unitarity
        np.testing.assert_array_almost_equal(F.dot(F.T), np.eye(self.J))
        return F

    def transform(self, x):
        ''' Discrete real fourier transform 
        
        The decomposition order is such that the first element is 
        the constant term followed by sine and cosine terms. 

        :Parameters:
            x : numpy.ndarray
                signal
        '''
        return np.dot(self.F, x)

    def inverse(self, sp):
        ''' Inverse fourier transform

        :Parameters:
            sp : numpy.ndarray
                spectrum
        '''
        return np.dot(self.F.T, sp)

    def ticks(self, nTicks=5, format='%.0f', units=1.):
        ''' Return a tuple of ``xticklabels``, ``xticks`` and corresponding 
        indexes for axe formatting.

        :Parameters:
            nTicks : int
                number of ticks
            format : str
                number format string
            units : d=float
                units divider
        '''
        indexes = list()
        ticks = list()
        ticklabels = list()
        for i in xrange(nTicks):
            idx = self.J/(nTicks-1)*i
            indexes.append(idx)
            ticks.append(self.x[idx])
            if self.x[idx] == 0:
                ticklabels.append(format%(self.x[idx]/units))
            else:
                ticklabels.append(format%(self.x[idx]/units))
        return (ticklabels, ticks, indexes)
