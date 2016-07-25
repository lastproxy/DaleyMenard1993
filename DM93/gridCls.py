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
        fourier : None
            return direct Fourier transform matrix
        invFourier : None
            return inverse Fourier transform matrix
            
    '''
    
    def __init__(self, N, L):
        self.N=N
        self.L=L
        self.J = 2*self.N+1 


        self.k = np.array(range(self.N+1), dtype=float)
        self._allK = np.array(range(-self.N,self.N+1), dtype=float)

        self.x = np.array(  [   self.L*k/(2.*self.N +1.) 
                                for k in self._allK
                                ])
        self.F = self._fourier()

    def _fourier(self):
        F = np.zeros(shape=(self.J, self.J))
        
        F[:,0] = 1./np.sqrt(2.)
        for j in xrange(self.N+1):
            for n in xrange(1, self.N+1):
                F[j, 2*n-1]         = np.cos(2.*np.pi*n*self.x[j]/(self.L))
                F[j, 2*n]           = np.sin(2.*np.pi*n*self.x[j]/(self.L))
                F[self.N+j, 2*n-1]  = np.cos(2.*np.pi*n*self.x[self.N+j]/(self.L))
                F[self.N+j, 2*n]    = np.sin(2.*np.pi*n*self.x[self.N+j]/(self.L))

        F *= np.sqrt(2./self.J)
        return F
