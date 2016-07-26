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
        F : numpy.ndarray(float)
            Fourier transform matrix (unitary)

    :Methods:
        transform : numpy.ndarray(shape=self.J)
            return direct Fourier transform of signal
        inverse : numpy.ndarray(shape=self.J)
            return inverse Fourier transform of spectra
            
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
        ''' Fourier transform 
        
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
