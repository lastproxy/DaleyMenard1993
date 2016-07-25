''' Correlation models library '''
import numpy as np

class CorrModel(object):
    ''' Correlation model template class 
    <!> Not to be instantiated: for derivation purpose only <!>

    Derived class need to provide at least `_func()` method with 
    prototype::

        def _func(self, r, Lp)

    where ``r`` is a semi-positive float, the distance, and ``Lp`` 
    a model-specific length parameter (it is not the correlation length).

    Optionnaly, one could also override `powSpecTh()` which is meant
    to return the theoritical power spectrum.

    '''

    def __init__(self, grid, Lc):
        self.grid = grid
        self.Lc = Lc
        self.eFold = self._findEFold()
        self.Lp = self.Lc/self.eFold

    def corrFunc(self):
        f = np.vectorize(self._func)
        return f(self.grid.x, self.Lp)

    def powSpecTh(self): 
        raise NotImplementedError()

    def powSpecNum(self): 
        tf = self.grid.transform(self.corrFunc())
        # -- keep semi-positive wavenumbers only
        tf = tf[self.grid.N:]
        sp = tf ** 2
        # -- normalize power spectrum
        sp /= (sp[0]+ 2.*sum(sp[1:]))
        return sp

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
        raise NotImplementedError()
        

class Uncorrelated(CorrModel):
    ''' Uncorrelated model '''
    def __init__(self, grid, **kw):
        self.grid = grid
        self.Lp = 0.
        self.eFold = 0.
        self.Lc = 0.

    def _func(self, x, Lp):
        if x == 0:
            return 1.
        else:
            return 0.

    def powSpecTh(self):
        return np.ones(self.grid.halfK.shape)/self.grid.J

class Foar(CorrModel):
    ''' First order autoregressive correlation model '''
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
    def _func(self, x, Lp):
        return np.exp(-x**2/(2.*Lp**2))
        
    def powSpecTh(self):
        q = 2.*np.pi*self.grid.halfK/self.grid.L
        sp = np.exp(-q**2*self.Lp**2/2)
        sp /= (sp[0]+ 2.*sum(sp[1:]))
        return sp
