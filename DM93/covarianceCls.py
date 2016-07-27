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

''' Correlation models library '''
import numpy as np

class Covariance(object):
    ''' Covariance
    
    :Attributes:
        grid : `Grid`
            space domain descriptor
        matrix : np.ndarray
            symetric matrix

    :Methods:
        random : None|float, None|float
            generate a random realisation from the covariance model
    '''

    def __init__(self, grid, matrix):
        self.grid = grid

        # -- assert matrix is symetric
        for i in xrange(self.grid.J):
            for j in xrange(i):
                np.testing.assert_almost_equal(matrix[i,j], matrix[j,i])
        self.matrix = matrix

    def random(self, bias=0.):
        ''' Generate a random realisation from the covariance model
        
        :Parameters:
            bias : float
                uniform bias (mean)
        '''
        mean = bias * np.ones(self.grid.J)
        return np.random.multivariate_normal(mean, self.matrix)
        

class CorrModel(Covariance):
    ''' 

    :Attributes:
        name : str
            correlation model name
        grid : `Grid`
            space domain descriptor
        Lc : float
            correlation length (define as the distance where
            the correlation is less or equal to e**-0.5)
        Lp : float
            correlation model specific length parameter
        eFold : float
            ratio Lc/Lp
        matrix : np.ndarray
            correlation matrix

    :Methods:
        powSpecTh : None
            return analytical power spectrum derived using the infinite
            domain approximation
        random : None|float, None|float
            generate a random realisation from the covariance model
    '''

    #--------------------------------------------
    # <!> Template class : do not instantiate
    #   use for derivation only
    #--------------------------------------------

    name = None
    def __init__(self, grid, Lc):
        self.grid = grid
        self.Lc = Lc
        self.eFold = self._findEFold()
        self.Lp = self.Lc/self.eFold
        self.matrix = self._buildMatrix()

    def corrFunc(self):
        f = np.vectorize(self._func)
        return f(self.grid.x, self.Lp)

    def powSpecTh(self): 
        raise NotImplementedError()

    def _findEFold(self, maxR=3., res=1000):
        f0 = self._func(0, 1.)
        eFold = None
        while eFold == None:
            r = np.linspace(0., maxR, res)
            dom = np.where(self._func(r,1.) <= f0/np.sqrt(np.e))[0]
            if len(dom) == 0:
                maxR += 1.
            else: 
                eFold = r[np.min(dom)]
        return eFold

    def _func(self, r, Lp):
        raise NotImplementedError('Template class: not to be instantiated')
        
    def _buildMatrix(self):
        g = self.grid
        C = np.eye(g.J)
        for i in xrange(g.J):
            for j in xrange(i):
                d = np.abs(g.x[i]-g.x[j])
                if d > g.L/2.: d = g.L - d
                C[i,j] = self._func(d, self.Lp)
                C[j,i] = C[i,j]
        return C

        

class Uncorrelated(CorrModel):
    ''' Uncorrelated model ''' 
    __doc__ += CorrModel.__doc__
    name = 'uncorrelated'

    def __init__(self, grid, *args):
        self.grid = grid
        self.Lp = 0.
        self.eFold = 0.
        self.Lc = 0.
        self.matrix = np.eye(self.grid.J)

    def _func(self, x, Lp):
        if x == 0:
            return 1.
        else:
            return 0.

    def powSpecTh(self):
        return np.ones(self.grid.halfK.shape)/self.grid.J

class Foar(CorrModel):
    ''' First order autoregressive correlation model '''
    __doc__ += CorrModel.__doc__
    name = 'foar'

    def _func(self, x, Lp):
        x = np.abs(x)/Lp
        return np.exp(-x)
        
    def powSpecTh(self):
        q = 2.*np.pi*self.grid.halfK/self.grid.L
        sp = (1. + q**2*self.Lp**2)**-1
        sp /= (sp[0]+ 2.*sum(sp[1:]))
        return sp

class Soar(CorrModel):
    ''' Second order autoregressive correlation model '''
    __doc__ += CorrModel.__doc__
    name = 'soar'

    def _func(self, x, Lp):
        x = np.abs(x)/Lp
        return (1.+ x)*np.exp(-x)
        
    def powSpecTh(self):
        q = 2.*np.pi*self.grid.halfK/self.grid.L
        sp = (1. + q**2*self.Lp**2)**-2
        sp /= (sp[0]+ 2.*sum(sp[1:]))
        return sp

class Gaussian(CorrModel):
    ''' Gaussian correlation model '''
    __doc__ += CorrModel.__doc__
    name = 'gaussian'

    def _func(self, x, Lp):
        return np.exp(-x**2/(2.*Lp**2))
        
    def powSpecTh(self):
        q = 2.*np.pi*self.grid.halfK/self.grid.L
        sp = np.exp(-q**2*self.Lp**2/2)
        sp /= (sp[0]+ 2.*sum(sp[1:]))
        return sp
