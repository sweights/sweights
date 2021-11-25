## Demonstrates some examples of the package

## external requirements
import os
from argparse import ArgumentParser
import numpy as np
from scipy.stats import norm, expon, uniform
from scipy.stats import kendalltau
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import ExtendedUnbinnedNLL
from iminuit.pdg_format import pdg_format
import boost_histogram as bh

## from this code
from sweights import sweight, cow, cov_correct, approx_cov_correct, kendall_tau

# make a toy model

Ns = 5000
Nb = 5000
ypars = [Ns,Nb]

# mass
mrange = (0,1)
mu = 0.5
sg = 0.1
lb = 10
mpars = [mu,sg,lb]

# decay time
trange = (0,1)
tlb = 2
tpars = [tlb]

# generate the toy
def generate(Ns,Nb,mu,sg,lb,tlb,poisson=False,ret_true=False):

  Nsig = np.random.poisson(Ns) if poisson else Ns
  Nbkg = np.random.poisson(Nb) if poisson else Nb

  sigM = norm(mu,sg)
  bkgM = expon(mrange[0], lb)

  sigT = expon(trange[0], tlb)
  bkgT = uniform(trange[0],trange[1]-trange[0])

  # generate
  sigMflt = sigM.cdf(mrange)
  bkgMflt = bkgM.cdf(mrange)
  sigTflt = sigT.cdf(trange)
  bkgTflt = bkgT.cdf(trange)

  sigMvals = sigM.ppf( np.random.uniform(*sigMflt,size=Nsig) )
  sigTvals = sigT.ppf( np.random.uniform(*sigTflt,size=Nsig) )

  bkgMvals = bkgM.ppf( np.random.uniform(*bkgMflt,size=Nbkg) )
  bkgTvals = bkgT.ppf( np.random.uniform(*bkgTflt,size=Nbkg) )

  Mvals = np.concatenate( (sigMvals, bkgMvals) )
  Tvals = np.concatenate( (sigTvals, bkgTvals) )

  truth = np.concatenate( ( np.ones_like(sigMvals), np.zeros_like(bkgMvals) ) )

  if ret_true:
    return np.stack( (Mvals,Tvals,truth), axis=1 )
  else:
    return np.stack( (Mvals,Tvals), axis=1 )

# define mass pdf for plotting etc.
def mpdf(x, Ns, Nb, mu, sg, lb, comps=['sig','bkg']):

  sig  = norm(mu,sg)
  sigN = np.diff( sig.cdf(mrange) )

  bkg  = expon(mrange[0], lb)
  bkgN = np.diff( bkg.cdf(mrange) )

  tot = 0
  if 'sig' in comps: tot += Ns * sig.pdf(x) / sigN
  if 'bkg' in comps: tot += Nb * bkg.pdf(x) / bkgN

  return tot

# define mass pdf for iminuit fitting
def mpdf_min(x, Ns, Nb, mu, sg, lb):
  return (Ns+Nb, mpdf(x, Ns, Nb, mu, sg, lb) )

# define time pdf for plotting etc.
def tpdf(x, Ns, Nb, tlb, comps=['sig','bkg']):

  sig  = expon(trange[0],tlb)
  sigN = np.diff( sig.cdf(trange) )

  bkg  = uniform(trange[0],trange[1]-trange[0])
  bkgN = np.diff( bkg.cdf(trange) )

  tot = 0
  if 'sig' in comps: tot += Ns * sig.pdf(x) / sigN
  if 'bkg' in comps: tot += Nb * bkg.pdf(x) / bkgN

  return tot

# define signal only time pdf for cov corrector
def tpdf_cor(x, tlb):
  return tpdf(x,1,0,tlb,['sig'])

def myerrorbar(data, ax, bins, range, wts=None, label=None, col=None):
  col = col or 'k'
  nh, xe = np.histogram(data,bins=bins,range=range)
  cx = 0.5*(xe[1:]+xe[:-1])
  err = nh**0.5
  if wts is not None:
    whist = bh.Histogram( bh.axis.Regular(bins,*range), storage=bh.storage.Weight() )
    whist.fill( data, weight = wts )
    cx = whist.axes[0].centers
    nh = whist.view().value
    err = whist.view().variance**0.5

  ax.errorbar(cx, nh, err, capsize=2,label=label,fmt=f'{col}o')

def plot(toy, draw_pdf=True, save=None):

  nbins = 50

  fig, ax = plt.subplots(1,2,figsize=(12,4))

  myerrorbar(toy[:,0],ax[0],bins=nbins,range=mrange)
  myerrorbar(toy[:,1],ax[1],bins=nbins,range=trange)

  if draw_pdf:
    m = np.linspace(*mrange,400)
    mN = (mrange[1]-mrange[0])/nbins

    bkgm = mpdf(m, *(ypars+mpars),comps=['bkg'])
    sigm = mpdf(m, *(ypars+mpars),comps=['sig'])
    totm = bkgm + sigm

    ax[0].plot(m, mN*bkgm, 'r--', label='Background')
    ax[0].plot(m, mN*sigm, 'g:' , label='Signal')
    ax[0].plot(m, mN*totm, 'b-' , label='Total PDF')

    t = np.linspace(*trange,400)
    tN = (trange[1]-trange[0])/nbins

    bkgt = tpdf(t, *(ypars+tpars),comps=['bkg'])
    sigt = tpdf(t, *(ypars+tpars),comps=['sig'])
    tott = bkgt + sigt

    ax[1].plot(t, tN*bkgt, 'r--', label='Background')
    ax[1].plot(t, tN*sigt, 'g:' , label='Signal')
    ax[1].plot(t, tN*tott, 'b-' , label='Total PDF')

  ax[0].set_xlabel('Mass')
  ax[0].set_ylim(bottom=0)
  ax[0].legend()

  ax[1].set_xlabel('Time')
  ax[1].set_ylim(bottom=0)
  ax[1].legend()

  fig.tight_layout()
  if save: fig.savefig(save)

def plot_wts(x, sw, bw, ylabel='Weight',save=None):

  fig,ax = plt.subplots()
  ax.plot(x, sw, 'b--', label='Signal')
  ax.plot(x, bw, 'r:' , label='Background')
  ax.plot(x, sw+bw, 'k-', label='Sum')
  ax.set_xlabel('Mass')
  ax.set_ylabel('Weight')
  fig.tight_layout()
  if save: fig.savefig(save)

def plot_tweighted(x, wts, wtnames=[], funcs=[], save=None):

  fig, ax = plt.subplots()

  t = np.linspace(*trange,400)
  N = (trange[1]-trange[0])/50

  for i, wt in enumerate(wts):
    label = None
    if i<len(wtnames): label = wtnames[i]
    myerrorbar(x, ax, bins=50, range=trange, wts=wt, label=label, col=f'C{i}')
    if i<len(funcs):
      ax.plot(t,N*funcs[i](t),f'C{i}-')

  ax.legend()
  ax.set_xlabel('Time')
  ax.set_ylabel('Weighted Events')
  fig.tight_layout()
  if save: fig.savefig(save)

def wnll(tlb, tdata, wts):
  sig  = expon(trange[0],tlb)
  sigN = np.diff( sig.cdf(trange) )
  return -np.sum( wts * np.log( sig.pdf( tdata ) / sigN ) )

if __name__ == '__main__':

  parser = ArgumentParser()
  parser.add_argument('-p','--makeplots'  , default=False, action='store_true', help='Make plots')
  parser.add_argument('-d','--plotdir'    , default='__plots__'               , help='Save location for plots')
  parser.add_argument('-s','--seed'       , default=None, type=int            , help='Set random seed')
  parser.add_argument('-i','--interactive', default=False, action='store_true', help='Show plots interactively at the end')
  parser.add_argument('-v','--verbose'    , default=False, action='store_true', help='Print more output')
  args = parser.parse_args()

  if args.seed:
    np.random.seed(args.seed)

  if args.makeplots and not os.path.exists(args.plotdir):
    os.system(f'mkdir -p {args.plotdir}')

  ## generate the toy
  toy = generate(Ns,Nb,mu,sg,lb,tlb,ret_true=True)
  if args.makeplots: plot(toy, save=f'{args.plotdir}/toy.png')

  ## print kendall rank coeff
  kts = kendall_tau(toy[:,0],toy[:,1])
  print('Kendall Tau:', pdg_format( kts[0], kts[1] ) )

  ## fit the toy mass
  mi = Minuit( ExtendedUnbinnedNLL(toy[:,0], mpdf_min), Ns=Ns, Nb=Nb, mu=mu, sg=sg, lb=lb )
  mi.limits['Ns'] = (0,Ns+Nb)
  mi.limits['Nb'] = (0,Ns+Nb)
  mi.limits['mu'] = mrange
  mi.limits['sg'] = (0,mrange[1]-mrange[0])
  mi.limits['lb'] = (0,50)

  mi.migrad()
  mi.hesse()
  if args.verbose: print(mi)

  # define estimated functions
  spdf = lambda m: mpdf(m,*mi.values,comps=['sig'])
  bpdf = lambda m: mpdf(m,*mi.values,comps=['bkg'])

  # make the sweighter
  print('Compute sweights')
  sweighter = sweight( toy[:,0], [spdf,bpdf], [mi.values['Ns'],mi.values['Nb']], (mrange,), method='summation', compnames=('sig','bkg'), verbose=args.verbose, checks=args.verbose )

  # get the COW equivalents (they should be normalised but the cow should also take care if they are not)
  gs = lambda m: mpdf(m,*mi.values,comps=['sig']) / mi.values['Ns']
  gb = lambda m: mpdf(m,*mi.values,comps=['bkg']) / mi.values['Nb']
  Im = 1 #lambda m: mpdf(m,*mi.values) / (mi.values['Ns'] + mi.values['Nb'] )

  # make the cow
  print('Compute cow weights')
  cw = cow(mrange, spdf, gb, Im, verbose=args.verbose)

  # compare the two
  flbs = []
  for meth, cls in zip( ['SW','COW'], [sweighter,cw] ):

    # plot weights
    x = np.linspace(*mrange,400)
    swp = cls.getWeight(0,x)
    bwp = cls.getWeight(1,x)
    if args.makeplots: plot_wts(x, swp, bwp, save=f'{args.plotdir}/{meth}_wts.png')

    # fit weighted data
    wts = cls.getWeight(0,toy[:,0])
    nll = lambda tlb: wnll(tlb, toy[:,1], wts)

    # do the minimisation
    tmi = Minuit( nll, tlb=tlb )
    tmi.limits['tlb'] = (1,3)
    tmi.errordef = Minuit.LIKELIHOOD
    tmi.migrad()
    tmi.hesse()

    # and do the correction
    fval = np.array(tmi.values)
    flbs.append(fval[0])
    fcov = np.array( tmi.covariance.tolist() )

    # first order correction
    ncov = approx_cov_correct(tpdf_cor, toy[:,1], wts, fval, fcov, verbose=args.verbose)

    # second order correction
    hs  = tpdf_cor
    ws  = lambda m: cls.getWeight(0,m)
    W   = cls.Wkl

    # these derivatives can be done numerically but for the sweights / COW case it's straightfoward to compute them
    ws = lambda Wss, Wsb, Wbb, gs, gb: (Wbb*gs - Wsb*gb) / ((Wbb-Wsb)*gs + (Wss-Wsb)*gb)
    dws_Wss = lambda Wss, Wsb, Wbb, gs, gb: gb * ( Wsb*gb - Wbb*gs ) / (-Wss*gb + Wsb*gs + Wsb*gb - Wbb*gs)**2
    dws_Wsb = lambda Wss, Wsb, Wbb, gs, gb: ( Wbb*gs**2 - Wss*gb**2 ) / (Wss*gb - Wsb*gs - Wsb*gb + Wbb*gs)**2
    dws_Wbb = lambda Wss, Wsb, Wbb, gs, gb: gs * ( Wss*gb - Wsb*gs ) / (-Wss*gb + Wsb*gs + Wsb*gb - Wbb*gs)**2

    tcov = cov_correct(hs, [gs,gb], toy[:,1], toy[:,0], wts, [mi.values['Ns'],mi.values['Nb']], fval, fcov, [dws_Wss,dws_Wsb,dws_Wbb],[W[0,0],W[0,1],W[1,1]], verbose=args.verbose)

    if args.verbose: print('Method:', meth, f'- covariance corrected {fval[0]:.1f} +/- {fcov[0,0]**0.5:.1f} ---> {fval[0]:.1f} +/- {tcov[0,0]**0.5:.1f}')

  ## plot weight T distribution
  swf  = lambda t: tpdf(t, mi.values['Ns'], 0, flbs[0], comps=['sig'] )
  cowf = lambda t: tpdf(t, mi.values['Ns'], 0, flbs[1], comps=['sig'] )
  sws  = sweighter.getWeight(0, toy[:,0])
  scow = cw.getWeight(0, toy[:,0])

  if args.makeplots: plot_tweighted(toy[:,1], [sws,scow], ['SW','COW'], funcs=[swf,cowf], save=f'{args.plotdir}/tfit.png' )


  if args.interactive: plt.show()
