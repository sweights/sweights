from scipy.stats import uniform
from scipy.integrate import quad
from scipy import linalg
import numpy as np

class cow():
  def __init__(self, mrange, gs, gb, Im=1, obs=None, renorm=True):
    '''
    mrange: a two element tuple for the integration range in mass
    gs:     a function for the signal (numerator) must accept a single argument in this case mass
    gb:     a function or list of functions for the backgrond (numerator) must accept a single argument in this case mass
    Im:     a function to evaluate the I(m) (denominator) must accept a single argument in this case mass (default is uniform 1)
    obs:    you can pass the observed distribution to evaluate Im instead. this expects the weights and bin edges in a two element tuple
            like the return value of np.histogram
    renorm: renormalise passed functions to unity (you can override this if you already know it's true)
    '''
    self.renorm = renorm
    self.mrange = mrange
    self.gs = self.normalise(gs)
    self.gb = [ self.normalise(g) for g in gb ] if hasattr(gb,'__iter__') else [ self.normalise(gb) ]
    self.gk = [self.gs] + self.gb
    if Im == 1:
      un = uniform(*mrange)
      n  = np.diff( un.cdf(mrange) )
      self.Im = lambda m: un.pdf(m) / n
    else:
      self.Im = self.normalise(Im)

    self.obs = obs
    if obs:
      if len(obs)!=2: raise ValueError('The observation must be passed as length two object containing weights and bin edges (w,xe) - ie. what is returned by np.histogram()')
      w, xe = obs
      if len(w)!=len(xe)-1: raise ValueError('The bin edges and weights do not have the right respective dimensions')
      # normalise
      w = w/np.sum(w)  # sum of wts now 1
      w /= (mrange[1]-mrange[0])/len(w) # now divide by bin width to get a function which integrates to 1
      f = lambda m: w[ np.argmin( m >= xe )-1 ]
      self.Im = np.vectorize(f)

    # compute Wkl matrix
    self.Wkl = self.compWkl()
    print('Found Wkl Matrix:')
    print( '\t', str(self.Wkl).replace('\n','\n\t ') )
    # invert for Akl matrix
    self.Akl = linalg.solve( self.Wkl, np.identity( len(self.Wkl) ), assume_a='pos' )
    print('Found Akl Matrix:')
    print( '\t', str(self.Akl).replace('\n','\n\t ') )

  def normalise(self, f):
    if self.renorm:
      N = quad(f,*self.mrange)[0]
      return lambda m: f(m) / N
    else:
      return f

  def compWklElem(self, k, l):
    # check it's available in m
    assert( k < len(self.gk) )
    assert( l < len(self.gk) )

    def integral(m):
      return self.gk[k](m) * self.gk[l](m) / self.Im(m)

    if self.obs is None:
      return quad( integral, self.mrange[0], self.mrange[1] )[0]
    else:
      tint = 0
      xe = self.obs[1]
      for le, he in zip(xe[:-1],xe[1:]):
        tint += quad( integral, le, he )[0]
      return tint

  def compWkl(self):

    n = len(self.gk)

    ret = np.identity(n)

    for i in range(n):
      for j in range(n):
        if i>j: ret[i,j] = ret[j,i]
        else:   ret[i,j] = self.compWklElem(i,j)

    return ret

  def wk(self, k, m):
    n = len(self.gk)
    return np.sum( [ self.Akl[k,l] * self.gk[l](m) / self.Im(m) for l in range(n) ], axis=0 )

  def getWeight(self, k, m):
    return self.wk(k,m)



