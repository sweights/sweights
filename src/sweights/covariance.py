# implements a covariance correction for weighted data fits
import numpy as np
from scipy.misc import derivative

## derivative of function pdf with respect to variable at index var
## evaluated at point point
def partial_derivative(pdf, var, point, data):
  args = point[:]
  def wraps(x):
    args[var] = x
    return pdf(data,*args)
  return derivative(wraps,point[var], dx=1e-6)

# you should pass the pdf function
# which must be in the form pdf(data,*pars) e.g. a 1D Gaussian would be pdf(x,mean,sigma)
# then pass the data (should be an appropriate degree numpy array to be passed to your pdf
# then pass the weights (should have same shape as data)
# then pass the fit values and fitted covariance of the nominal fit

def approx_cov_correct(pdf, data, wts, fvals, fcov, verbose=False):

  dim = len(fvals)
  assert(fcov.shape[0]==dim and fcov.shape[1]==dim)

  Djk = np.zeros(fcov.shape)

  prob = pdf(data,*fvals)
  #print(prob)
  for j in range(dim):
    derivj = partial_derivative(pdf,j,fvals,data)
    for k in range(dim):
      derivk = partial_derivative(pdf,k,fvals,data)

      Djk[j,k] = np.sum( wts**2 * (derivj*derivk) / prob**2 )

  corr_cov = fcov * Djk * fcov.T

  if verbose:
    print('First order covariance correction for weighted events')
    print('  Original covariance:')
    print('\t', str(fcov).replace('\n','\n\t '))
    print('  Corrected covariance:')
    print('\t', str(corr_cov).replace('\n','\n\t '))

  return corr_cov

def cov_correct(hs, gxs, hdata, gdata, weights, Nxs, fvals, fcov, dw_dW_fs, Wvals, verbose=False):

  dim_kl = len(fvals)
  assert(fcov.shape[0]==dim_kl and fcov.shape[1]==dim_kl )

  dim_xy = len(gxs)
  assert( len(Nxs)==dim_xy )
  dim_E = int(dim_xy*(dim_xy+1)/2)  # indepdent elements of symmetric Wxy matrix is n(n+1)/2
  assert( len(dw_dW_fs)==dim_E )
  assert( len(Wvals)==dim_E )

  HHpH_term = approx_cov_correct(hs, hdata, weights, fvals, fcov, verbose=False)

  # now construct the E and C' matrices
  Ekl = np.empty((dim_kl,dim_E))
  for l in range(dim_kl):
    derivl = partial_derivative(hs,l,fvals,hdata)
    gxevs = [ gx(gdata) for gx in gxs ]
    for xy in range(dim_E):
      Ekl[l,xy] = np.sum( dw_dW_fs[xy]( *Wvals, *gxevs ) * derivl )

  Ckl = np.empty((dim_E,dim_E))
  gtot = np.sum( [ Nxs[i]*gxs[i](gdata) for i in range(dim_xy) ], axis=0 )

  Citer = [ (gxs[i], gxs[j]) for i in range(dim_xy) for j in range(dim_xy) if i>=j ]

  for i, gis in enumerate(Citer):
    for j, gjs in enumerate(Citer):
      Ckl[i,j] = np.sum( gis[0](gdata) * gis[1](gdata) * gjs[0](gdata) * gjs[1](gdata) / gtot**4 )

  HECEH_term = fcov @ ( Ekl @ Ckl @ Ekl.T ) @ fcov.T

  tcov = HHpH_term - HECEH_term

  if verbose:
    print('Full covariance correction for weighted events')
    print('  Original covariance:')
    print('\t', str(fcov).replace('\n','\n\t '))
    print('  Corrected covariance:')
    print('\t', str(tcov).replace('\n','\n\t '))

  return tcov
