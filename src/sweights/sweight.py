# vim: ts=4 sw=4 et

import numpy as np
from scipy.integrate import quad, nquad
from scipy.linalg import solve
from scipy.interpolate import InterpolatedUnivariateSpline
import matplotlib.pyplot as plt

class sweight():

    ### Need to pass:
    ###    data:          must be a numpy ndarray with shape (nevs,ndiscvar)
    ###    discvarranges: pass a range for the discriminating variables in which the pdfs are to be normalised
    ###    pdfs:          a list of the component pdfs (these should be simple python functions or ROOT.RooAbsPdf if running with the 'roofit' method)
    ###                   #TODO allow passing list of RooFit pdfs for other sweight methods
    ###    yields:        a list of the component yields

    def __init__(self,
                 data,
                 pdfs=[],
                 yields=[],
                 discvarranges=None,
                 method='summation',
                 compnames=None,
                 alphas=None,
                 rfobs=[],
                 verbose=True,
                 checks=True
                ):

        self.allowed_methods = ['summation','integration','refit','subhess','tsplot','roofit']
        self.method = method
        if self.method not in self.allowed_methods:
            raise RuntimeError( self.method,'is not an allowed method. Must be one of', self.allowed_methods )

        if self.method=='refit':
          try:
            import iminuit
            from iminuit import Minuit
          except:
            raise RuntimeError( 'To run sweight with refit method iminuit must be installed')
          if iminuit.version.version.split('.')[0]=='1':
            raise RuntimeError( 'iminuit version must be > 2.x' )

        self.verbose = verbose
        if self.verbose: print('Initialising SWeighter with the', self.method, 'method:')

        # get data in right format
        self.data = data
        if len(self.data.shape)==1:
            self.data = self.data.reshape( len(self.data), 1 )
        if not len(self.data.shape)==2:
            raise ValueError( 'Input data is in the wrong format. Should be a numpy ndarray with shape (nevs,ncomps)' )
        self.nevs = self.data.shape[0]
        self.ndiscvars = self.data.shape[1]

        # sort out the disciminant variables integration ranges
        self.discvarranges = discvarranges
        if self.discvarranges is None:
            self.discvarranges = tuple( tuple(-np.inf,np.inf) for i in range(self.ndiscvars) )
        if not (self.ndiscvars==len(self.discvarranges)):
            raise ValueError( 'You dont seemed to have passed sufficient ranges', len(self.discvarranges),'for the number of discriminant variables', len(self.ndiscvars))
        self.discvarranges = np.array( self.discvarranges )

        # remove events out of range
        self.data = self.data[ np.logical_and( self.data >= self.discvarranges.T[0], self.data <= self.discvarranges.T[1] ).all( axis=1 ) ]

        # check pdfs and write normalisation terms
        self.pdfs = pdfs
        self.yields = yields
        if not (len(self.yields)==len(self.pdfs)):
          raise ValueError( 'The number of yields is not the same as the number of pdfs' )

        self.ncomps = len(self.yields)
        if self.method != 'roofit':
          self.pdfnorms = [ nquad( pdf, self.discvarranges )[0] for pdf in self.pdfs ]
          if checks:
            print('    PDF normalisations:')
            for i, norm in enumerate(self.pdfnorms):
              print('\t', i, norm)
          self.pdfsum   = lambda x: sum( [self.yields[i]*self.pdfs[i](x)/self.pdfnorms[i] for i in range(self.ncomps)] )

        # if subhess method check alpha has been passed
        self.alphas = alphas
        if self.method=='subhess':
            if alphas is None:
                raise RuntimeError('If using method subhess you must pass the covariance (alpha) matrix')
        if alphas is not None and not alphas.shape==np.zeros( (self.ncomps,self.ncomps) ).shape:
            raise RuntimeError('You have passed an alpha matrix but it has the wrong shape')

        # add names if needed
        self.compnames = compnames
        if self.compnames is None:
          self.compnames = [ str(i) for i in range(self.ncomps) ]

        # compute the W matrix
        self.computeWMatrix()

        # solve the alpha matrix
        self.solveAlphas()

        # do the tsplot implementation if asked
        self.tsplotweights = None
        self.tsplotw = None
        if self.method=='tsplot':
          self.tsplotweights = self.runTSPlot()
          self.tsplotw = [ InterpolatedUnivariateSpline( *self.tsplotweights[:,[0,i+1]].T, k=3 ) for i in range(self.ncomps) ]

        # do the RooFit implementation if asked
        self.roofitweights = None
        self.roofitw = None
        self.rfobs = rfobs
        if self.method=='roofit':
          if len(self.rfobs) != self.ndiscvars:
              raise RuntimeError('You must pass a list of RooFit observables the same length as the data when running with the roofit method')

          self.roofitweights = self.runRooFit()
          self.roofitw = [ InterpolatedUnivariateSpline( *self.roofitweights[:,[0,i+1]].T, k=3 ) for i in range(self.ncomps) ]

        # print checks
        if checks: self.printChecks()

    def computeWMatrix(self):
        self.Wkl = np.zeros( (self.ncomps,self.ncomps) )
        if self.method in ['refit','subhess','tsplot','roofit']:
          return self.Wkl
        for i, di in enumerate(self.pdfs):
            dinorm = self.pdfnorms[i]
            for j, dj in enumerate(self.pdfs):
                djnorm = self.pdfnorms[j]
                if j<i:
                    self.Wkl[i,j] = self.Wkl[j,i]
                else:
                    if self.method=='integration':
                        self.Wkl[i,j] = nquad( lambda *args: ( di(*args)/dinorm  * dj(*args)/djnorm ) / self.pdfsum(*args), self.discvarranges )[0]
                    elif self.method=='summation':
                        self.Wkl[i,j] = np.sum( ( di(*self.data.T)/dinorm * dj(*self.data.T)/djnorm ) / self.pdfsum(*self.data.T)**2 )
                    else:
                        self.Wkl[i,j] = 0.

        if self.method in ['integration','summation'] and self.verbose:
            print('    W-matrix:')
            print('\t'+ str(self.Wkl).replace('\n','\n\t'))

        return self.Wkl

    def nll(self, pars):
      assert(len(pars)==self.ncomps)
      nobs = sum(pars)
      nest = np.sum( np.log( sum( [ pars[i]*pdf(*self.data.T)/self.pdfnorms[i] for i, pdf in enumerate(self.pdfs) ] ) ) )
      return nobs - nest

    def solveAlphas(self):
        if self.method in ['integration','summation']:
            sol = np.identity( len(self.Wkl) )
            self.alphas = solve( self.Wkl, sol, assume_a='pos' )
        elif self.method in ['refit']:
            #mi = Minuit.from_array_func(self.nll, tuple(self.yields), errordef=Minuit.LIKELIHOOD, pedantic=False)
            mi = Minuit(self.nll, tuple(self.yields) )
            mi.errordef=Minuit.LIKELIHOOD
            mi.migrad()
            mi.hesse()
            self.alphas = np.array(mi.covariance.tolist())
        elif self.method in ['tsplot','roofit']:
            self.alphas = np.identity( len(self.Wkl) )
            return self.alphas

        if self.verbose:
            print('    alpha-matrix:')
            print('\t' + str(self.alphas).replace('\n','\n\t'))

        return self.alphas

    def getWeight(self,icomp=0, *args):
        if self.method == 'tsplot':
            return self.tsplotw[icomp](*args)
        elif self.method == 'roofit':
            return self.roofitw[icomp](*args)
        else:
            return sum( [self.alphas[i,icomp] * self.pdfs[i](*args)/self.pdfnorms[i] for i in range(self.ncomps)] ) / sum( [self.yields[i] * self.pdfs[i](*args)/self.pdfnorms[i] for i in range(self.ncomps)] )

    def makeWeightPlot(self, axis=None, dopts=['r','b','g','m','c','y'], labels=None):

        if self.ndiscvars!=1:
            print('WARNING - I dont know how to plot this')
            return None

        while len(dopts)<self.ncomps:
          dopts.append('b')

        ax = axis or plt.gca()

        x = np.linspace( *self.discvarranges[0], 400 )

        labels = labels or [ '$w_{{{0}}}$'.format(comp) for comp in self.compnames ]

        for comp in range(self.ncomps):
            ax.plot( x, self.getWeight(comp,x), color=dopts[comp], linewidth=2, label=labels[comp] )

        label = labels[-1] if len(labels)>self.ncomps else '$\sum_i w_{i}$'
        ax.plot( x, sum( [ self.getWeight(c,x) for c in range(self.ncomps) ] ), 'k-', linewidth=3, label=label)

        ax.legend()

    def printChecks(self):
        if self.method!='roofit':
            self.intws = np.identity( self.ncomps )
            for i in range(self.ncomps):
                for j in range(self.ncomps):
                    self.intws[i,j] = nquad( lambda *args:  self.getWeight(i,*args) * self.pdfs[j](*args) / self.pdfnorms[j], self.discvarranges )[0]
            print('    Integral of w*pdf matrix (should be close to the identity):')
            #with np.printoptions(precision=3, suppress=True):
            print( '\t' + str(self.intws).replace('\n','\n\t') )

        print('    Check of weight sums (should match yields):')
        self.sows = [ np.sum( self.getWeight(i,*self.data.T) ) for i in range(self.ncomps) ]
        header = '\t{:10s} | {:^10s} | {:^10s} | {:^9s} |'.format('Component','sWeightSum','Yield','Diff')
        print(header)
        print( '\t'+'-'.join( ['' for i in range(len(header)+1)]) )

        for i, yld in enumerate(self.yields):
            print('\t  {:<8d} | {:10.4f} | {:10.4f} | {:8.2f}% |'.format(i, self.sows[i], yld, 100.*(yld-self.sows[i])/self.sows[i]))


    def runTSPlot(self):

        if self.method!='tsplot':
            raise RuntimeError('Method is',self.method,'but calling the tsplot function.')

        # this works very differently for nD fits
        # so will not implement it here
        if self.ndiscvars!=1:
            raise RuntimeError('Sorry but I can\'t do the tsplot method for >1D fits')

        # make the tree for TSPlot

        # temporary datfile
        datfile = open('.data.dat','w')
        for x in self.data:
          datfile.write('{}'.format(x[0]))
          for i, pdf in enumerate(self.pdfs):
            x_pdf = pdf(x[0])/self.pdfnorms[i]
            datfile.write(' {}'.format(x_pdf))
          datfile.write('\n')
        datfile.close()

        # now make the tree and draw some plots to check it
        import ROOT as r
        r.gROOT.SetBatch()

        tree = r.TTree('data','data')
        read_str = "x/D:" + ":".join( ["fx_%s"%a for a in self.compnames] )

        tree.ReadFile('.data.dat', read_str, ' ')
        c = r.TCanvas("c","c",(self.ncomps+1)*600,400)
        c.Divide(self.ncomps+1,1)
        c.cd(1)
        tree.Draw("x")
        for i in range(self.ncomps):
          c.cd(i+2)
          tree.Draw("fx_%s:x"%self.compnames[i])
        c.Update()
        c.Modified()
        c.Draw()
        r.gErrorIgnoreLevel = r.kInfo
        c.Print('figs/tree.pdf')

        # now do the TSPlot
        from array import array
        tsplot = r.TSPlot(0, self.ndiscvars, len(self.data), self.ncomps, tree)
        sel_str = "x:" + ":".join( ["fx_%s"%a for a in self.compnames] )
        tsplot.SetTreeSelection(sel_str)
        ne = array('i', [int(yld) for yld in self.yields] )
        tsplot.SetInitialNumbersOfSpecies(ne)
        tsplot.MakeSPlot("Q")

        # get the weights out of TSPlot
        weights = np.ndarray( len(self.data)*self.ncomps )
        tsplot.GetSWeights(weights)
        weights = np.reshape( weights, (-1, self.ncomps) )
        data_w_weights = np.append( self.data, weights, axis=1 )
        sorted_data_w_weights = data_w_weights[np.argsort(data_w_weights[:,0])]

        return sorted_data_w_weights

    def runRooFit(self):

        if self.method!='roofit':
            raise RuntimeError('Method is',self.method,'but calling the roofit function.')

        import ROOT as r
        from ROOT import RooFit as rf
        from ROOT import RooStats as rs
        r.RooMsgService.instance().setGlobalKillBelow( rf.FATAL )
        r.gErrorIgnoreLevel = r.kInfo

        # sort out observables
        rf_obs  = r.RooArgList()
        for obs in self.rfobs:
            if not obs.InheritsFrom('RooAbsReal'):
                raise RuntimeError('Found an observable which does not inherit from RooAbsReal')
            rf_obs.add( obs )

        # make the roodataset and fill it
        rf_dset = r.RooDataSet( 'data', 'data', r.RooArgSet(rf_obs) )
        for row in self.data:
          for i in range(self.ndiscvars):
            rf_obs.at(i).setVal( row[i] )
          rf_dset.add( r.RooArgSet(rf_obs) )

        # pdfs should now be of the RooFit form
        rf_pdfs = r.RooArgList()
        for pdf in self.pdfs:
            if not pdf.InheritsFrom('RooAbsPdf'):
                raise RuntimeError('Found a pdf which does not inherit from RooAbsPdf.')
            #pdf.getParameters( rf_dset ).setAttribAll("Constant")
            rf_pdfs.add( pdf )

        # yields can still be numbers
        rfylds = [ r.RooRealVar('y_%s'%self.compnames[i], 'y_%s'%self.compnames[i], yld, 0., 2.5*yld ) for i, yld in enumerate(self.yields) ]
        rf_ylds = r.RooArgList()
        for yld in rfylds:
          rf_ylds.add( yld  )

        # now can create the pdf
        rf_totpdf = r.RooAddPdf( 'pdf', 'pdf', rf_pdfs, rf_ylds, False )

        # fit it (probably already been done)
        rf_totpdf.fitTo( rf_dset, rf.Extended(True), rf.PrintLevel(-1) )

        # plot it
        c = r.TCanvas()
        for obs in self.rfobs:
          pl = obs.frame()
          rf_dset.plotOn(pl, rf.Binning(50) )
          rf_totpdf.plotOn(pl)
          pl.Draw()
          c.Update()
          c.Modified()
          c.Draw()
          r.gErrorIgnoreLevel = r.kInfo
          c.Print('figs/rf_%s.pdf'%obs.GetName())

        # now get the sweights using RooStats
        rf_sp = rs.SPlot( 'sdata', 'sdata', rf_dset, rf_totpdf, rf_ylds )

        weights = np.zeros( (rf_dset.numEntries(), self.ncomps) )

        for ev in range(rf_dset.numEntries()):
          rfvals = rf_dset.get(ev)
          for i, obs in enumerate(self.rfobs):
              assert( abs(rfvals.getRealValue( obs.GetName() ) - self.data[ev][i])<1e-6 )
          for i, comp in enumerate(self.compnames):
            weights[ev,i] = rfvals.getRealValue( 'y_'+comp+'_sw' )

        data_w_weights = np.append( self.data, weights, axis=1 )
        sorted_data_w_weights = data_w_weights[np.argsort(data_w_weights[:,0])]

        return sorted_data_w_weights
