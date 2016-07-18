''' Correlation models library '''
import numpy as np

class CorrModel(object):
    ''' Correlation model template class '''
    def __init__(self, grid, Lp):
        self.grid = grid
        self.Lp = Lp

    def corrFunc(self):
        raise NotImplementedError()

    def powSpec(self): 
        raise NotImplementedError()



class Uncorrelated(CorrModel):
    ''' Uncorrelated model '''
    def __init__(self, grid, **kw):
        self.grid = grid
        self.Lp = 0.

    def corrFunc(self):
        x = np.zeros(self.grid.nGP)
        x[self.grid.N] = 1.
        return x

    def powSpec(self):
        return np.ones(self.grid.k.shape)/self.grid.nGP

class Foar(CorrModel):
    ''' First order autoregressive correlation model '''
    def corrFunc(self):
        x = np.abs(self.grid.x)/self.Lp
        return np.exp(-x)
        
    def powSpec(self):
        q = 2.*np.pi*self.grid.k/self.grid.L
        sp = (1. + q**2*self.Lp**2)
        sp /= (sp[0]+ 2.*sum(sp[1:]))
        return sp

class Soar(CorrModel):
    ''' Second order autoregressive correlation model '''
    def corrFunc(self):
        x = np.abs(self.grid.x)/self.Lp
        return (1.+ x)*np.exp(-x)
        
    def powSpec(self):
        q = 2.*np.pi*self.grid.k/self.grid.L
        sp = (1. + q**2*self.Lp**2)**-2
        sp /= (sp[0]+ 2.*sum(sp[1:]))
        return sp

class Gaussian(CorrModel):
    ''' Gaussian correlation model '''
    def corrFunc(self):
        return np.exp(-self.grid.x**2/(2.*self.Lp**2))
        
    def powSpec(self):
        q = 2.*np.pi*self.grid.k/self.grid.L
        sp = np.exp(-q**2*self.Lp**2/2)
        sp /= (sp[0]+ 2.*sum(sp[1:]))
        return sp
