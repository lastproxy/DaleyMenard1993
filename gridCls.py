''' Simple periodic grid class '''
import numpy as np 

class Grid(object):
    
    def __init__(self, N, L):
        self.N=N
        self.L=L
        self.nGP = 2*self.N+1 

        self.x = np.array(  [   self.L*j/(2.*self.N +1.) 
                                for j in xrange(-self.N, self.N+1)
                                ])
        self.k = np.array(range(self.N+1), dtype=float)
