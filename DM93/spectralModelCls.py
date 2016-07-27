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

class SpectralModel(object):
    ''' Simple 1D spectral model class

    '''
    def __init__(self, grid, dt):
        self.grid = grid
        self.dt = dt
        self.S = self._buildSpPropagator()
        self.M = self._buildGridPropagator() 

    def __call__(self, x):
        ''' Apply model propagator on state or matrix

        If `x` is a vector, M(x) = M.x
        if `x` is a matrix, M(x) == M.x.M'

        :Parameters:
            x : np.ndarray
                model state or matrix
        '''
        if x.ndim == 1:
            return self.M.dot(x)
        elif x.ndim == 2:
            return (self.M.dot(x)).dot(self.M.T)
        else:
            raise ValueError()


    def _buildSpPropagator(self):
        ''' build spectral propagator '''
        raise NotImplementedError()

    def _buildGridPropagator(self):
        ''' build grid propagator

        M = F.S.F' 
        '''
        return (self.grid.F.dot(self.S)).dot(self.grid.F.T)


class AdvectionDiffusionModel(SpectralModel):
    ''' Simple 1D advection + diffusion model 

    :Attributes:
        grid : `Grid`
            Periodic grid
        dt : float
            Time increment
        U : float
            Constant zonal wind speed [m/s]
        nu : float
            Viscosity coefficient [m/s]
        M : np.ndarray
            Grid space propagator
        S : np.ndarray
            Spectral space propagator
    
    Callable::
        
        x1 = modelInstance(x0)
    '''
    
    def __init__(self, grid, U, dt=1., nu=0):
        '''
        :Parameters:
            grid : `Grid`
                Periodic 1D grid
            U : float
                Constant zonal wind speed
            dt : float
                Time increment
            nu : float
                Viscosity coefficient [m/s]
        '''
        self.U = U
        self.nu = nu
        super(AdvectionDiffusionModel, self).__init__(grid, dt)

    def _buildSpPropagator(self):
        S = np.zeros(shape=(self.grid.J, self.grid.J))
        S[0,0] = 1.
        for j in xrange(1,self.grid.N+1):
            phi = 2.*np.pi*j*self.U*self.dt/self.grid.L
            ampl = np.exp(-4.*np.pi**2*self.nu*self.dt*j**2/self.grid.L**2)
            S[2*j-1, 2*j-1] = ampl * np.cos(phi)
            S[2*j, 2*j-1]   = ampl * np.sin(phi)
            S[2*j-1, 2*j]   = -ampl * np.sin(phi)
            S[2*j, 2*j]     = ampl * np.cos(phi)
        return S

        
