"""Implementation of the SWeight class."""

import numpy as np
from scipy.integrate import nquad
from scipy.linalg import solve
from scipy.interpolate import InterpolatedUnivariateSpline


class SWeight:
    """Produce sweights for a dataset given component pdfs."""

    def __init__(
        self,
        data,
        pdfs=[],
        yields=[],
        discvarranges=None,
        method="summation",
        compnames=None,
        alphas=None,
        rfobs=[],
        verbose=True,
        checks=True,
    ):
        """
        Initialize SWeight object.

        This will compute the W and A (or alpha) matrices which are used to
        produce the weight functions. Evaluation of these functions on a
        dataset is done in a different function :func:`get_weight`.

        Parameters
        ----------
        data : ndarray
            The dataset in the discriminating variable, should have shape
            (nevents,ndiscvars) so for one discriminating variable will just
            be (N,) or (N,1).
        pdfs : list of callable
            A list of the component pdfs. These should be simple python
            callable functions which are passed the same number of parameters
            as there are discriminating variables. If running with the
            'roofit' method (see `method`) then this can be a list of
            ROOT.RooAbsPdf objects. If you only have your pdfs defined as
            RooFit objects then you can use the wrapper function
            :func:`convert_rf_pdf` to convert them into the appropriate object
            type for this call and then use other methods.
        yields : list of float
            A list of the component yields.
        discvarranges : list of tuple or tuple of tuple, optional
            The ranges for each discriminating variable, as a tuple or list of
            two element tuples specifying the lower and upper bound for the
            range (the default is None which will use minus to plus infinity)
        method: str, optional
            The sweights method to use. Must be one of 'summation',
            'integration', 'subhess', 'tsplot', 'rootfit'. The recommended
            default is summation corresponding to Variant B of
            `arXiv:2112.04574 <https://arxiv.org/abs/2112.04574>`_
        compnames: list of str, optional
            A list of the component names. Only used for the legend entries
            when making a plot of the weight functions with
            :func:`make_weight_plot`
        alphas: ndarray, optional
            If using the 'subhess' method then the covariance matrix of a fit
            to the disciminanting variable(s) in which only the yields float
            must also be passed. This matrix is inverted to produce the
            W-matrix.
        rfobs: list, optional
            If using the 'roofit' method then the discriminating variables
            (of type RooRealVar) on which the pdfs depend must also be passed
        verbose: bool, optional
            Print output
        checks: bool, optional
            Perform some checks that the derived weights are self consistent,
            in particular check that the sum of all component weights for a
            given value of the discriminating variable(s) is unity, and check
            that the sum of all weights for a given component reproduces the
            yields. This will print additional output to the screen and issue
            warnings if checks are not passed.

        Notes
        -----
        Many of the arguments passed and methods implemented are not strictly
        necessary. In particular the 'tsplot' and 'roofit' methods are just
        wrappers for implementations elsewhere that were originally used as
        cross-checks. In future versions these two methods should be removed
        to simplify this call.

        See Also
        --------
        get_weight, make_weight_plot
        """
        self.allowed_methods = [
            "summation",
            "integration",
            "refit",
            "subhess",
            "tsplot",
            "roofit",
        ]
        self.method = method
        if self.method not in self.allowed_methods:
            raise RuntimeError(
                self.method,
                "is not an allowed method. Must be one of",
                self.allowed_methods,
            )

        if self.method == "refit":
            try:
                import iminuit
            except Exception:
                raise RuntimeError(
                    "To run with the 'refit' method iminuit must be installed"
                )
            if iminuit.version.version.split(".")[0] == "1":
                raise RuntimeError("iminuit version must be > 2.x")

        if self.method == "tsplot":
            try:
                import ROOT as r
            except Exception:
                raise RuntimeError("To run with 'tsplot method ROOT must be installed")
            if int(r.__version__.split(".")[0]) < 6:
                raise RuntimeError("ROOT version must be > 6.xx")

        if self.method == "roofit":
            try:
                import ROOT as r
            except Exception:
                raise RuntimeError(
                    """To run sweight with roofit method ROOT with RooFit must
                    be installed"""
                )
            if int(r.__version__.split(".")[0]) < 6:
                raise RuntimeError("ROOT version must be > 6.xx")

        self.verbose = verbose
        if self.verbose:
            print("Initialising sweight with the", self.method, "method:")

        # get data in right format
        self.data = data
        if len(self.data.shape) == 1:
            self.data = self.data.reshape(len(self.data), 1)
        if not len(self.data.shape) == 2:
            raise ValueError(
                """Input data is in the wrong format. Should be a numpy
                ndarray with shape (nevs,ncomps)"""
            )
        self.nevs = self.data.shape[0]
        self.ndiscvars = self.data.shape[1]

        # sort out the disciminant variables integration ranges
        self.discvarranges = discvarranges
        if self.discvarranges is None:
            self.discvarranges = tuple(
                tuple(-np.inf, np.inf) for i in range(self.ndiscvars)
            )
        if not (self.ndiscvars == len(self.discvarranges)):
            raise ValueError(
                "You dont seemed to have passed sufficient ranges",
                len(self.discvarranges),
                "for the number of discriminant variables",
                len(self.ndiscvars),
            )
        self.discvarranges = np.array(self.discvarranges)

        # remove events out of range
        self.data = self.data[
            np.logical_and(
                self.data >= self.discvarranges.T[0],
                self.data <= self.discvarranges.T[1],
            ).all(axis=1)
        ]

        # check pdfs and write normalisation terms
        self.pdfs = pdfs
        self.yields = yields
        if not (len(self.yields) == len(self.pdfs)):
            raise ValueError(
                "The number of yields is not the same as the number of pdfs"
            )

        self.ncomps = len(self.yields)
        if self.method != "roofit":
            self.pdfnorms = [nquad(pdf, self.discvarranges)[0] for pdf in self.pdfs]
            if checks:
                print("    PDF normalisations:")
                for i, norm in enumerate(self.pdfnorms):
                    print("\t", i, norm)
            self.pdfsum = lambda x: sum(
                [
                    self.yields[i] * self.pdfs[i](x) / self.pdfnorms[i]
                    for i in range(self.ncomps)
                ]
            )

        # if subhess method check alpha has been passed
        self.alphas = alphas
        if self.method == "subhess":
            if alphas is None:
                raise RuntimeError(
                    """If using method subhess you must pass the covariance
                    (alpha) matrix"""
                )
        if (
            alphas is not None
            and not alphas.shape == np.zeros((self.ncomps, self.ncomps)).shape
        ):
            raise RuntimeError(
                "You have passed an alpha matrix but it has the wrong shape"
            )

        # add names if needed
        self.compnames = compnames
        if self.compnames is None:
            self.compnames = [str(i) for i in range(self.ncomps)]

        # compute the W matrix
        self._compute_W_matrix()

        # solve the alpha matrix
        self._solve_alphas()

        # do the tsplot implementation if asked
        self.tsplotweights = None
        self.tsplotw = None
        if self.method == "tsplot":
            self.tsplotweights = self._run_tsplot()
            self.tsplotw = [
                InterpolatedUnivariateSpline(*self.tsplotweights[:, [0, i + 1]].T, k=3)
                for i in range(self.ncomps)
            ]

        # do the RooFit implementation if asked
        self.roofitweights = None
        self.roofitw = None
        self.rfobs = rfobs
        if self.method == "roofit":
            if len(self.rfobs) != self.ndiscvars:
                raise RuntimeError(
                    """You must pass a list of RooFit observables the same
                    length as the data when running with the roofit method"""
                )

            self.roofitweights = self._run_roofit()
            self.roofitw = [
                InterpolatedUnivariateSpline(*self.roofitweights[:, [0, i + 1]].T, k=3)
                for i in range(self.ncomps)
            ]

        # print checks
        if checks:
            self.print_checks()

    def _compute_W_matrix(self):
        self.Wkl = np.zeros((self.ncomps, self.ncomps))
        if self.method in ["refit", "subhess", "tsplot", "roofit"]:
            return self.Wkl
        for i, di in enumerate(self.pdfs):
            dinorm = self.pdfnorms[i]
            for j, dj in enumerate(self.pdfs):
                djnorm = self.pdfnorms[j]
                if j < i:
                    self.Wkl[i, j] = self.Wkl[j, i]
                else:
                    if self.method == "integration":
                        self.Wkl[i, j] = nquad(
                            lambda *args: (di(*args) / dinorm * dj(*args) / djnorm)
                            / self.pdfsum(*args),
                            self.discvarranges,
                        )[0]
                    elif self.method == "summation":
                        self.Wkl[i, j] = np.sum(
                            (di(*self.data.T) / dinorm * dj(*self.data.T) / djnorm)
                            / self.pdfsum(*self.data.T) ** 2
                        )
                    else:
                        self.Wkl[i, j] = 0.0

        if self.method in ["integration", "summation"] and self.verbose:
            print("    W-matrix:")
            print("\t" + str(self.Wkl).replace("\n", "\n\t"))

        return self.Wkl

    def _nll(self, pars):
        assert len(pars) == self.ncomps
        nobs = sum(pars)
        nest = np.sum(
            np.log(
                sum(
                    [
                        pars[i] * pdf(*self.data.T) / self.pdfnorms[i]
                        for i, pdf in enumerate(self.pdfs)
                    ]
                )
            )
        )
        return nobs - nest

    def _solve_alphas(self):
        if self.method in ["integration", "summation"]:
            sol = np.identity(len(self.Wkl))
            self.alphas = solve(self.Wkl, sol, assume_a="pos")
        elif self.method in ["refit"]:
            from iminuit import Minuit

            mi = Minuit(self._nll, tuple(self.yields))
            mi.errordef = Minuit.LIKELIHOOD
            mi.migrad()
            mi.hesse()
            self.alphas = np.array(mi.covariance.tolist())
        elif self.method in ["tsplot", "roofit"]:
            self.alphas = np.identity(len(self.Wkl))
            return self.alphas

        if self.verbose:
            print("    A-matrix:")
            print("\t" + str(self.alphas).replace("\n", "\n\t"))

        return self.alphas

    def get_weight(self, icomp=0, *args):
        """
        Return the weights.

        Get the weights for a given component and set of discriminating
        variable values.

        Parameters
        ----------
        icomp : int, optional
            Get the weight function for the ith component
        *args : ndarray
            The values of the discriminating variables (one argument for each)
            as numpy arrays

        Returns
        -------
        ndarray
            An array of the weights
        """
        if self.method == "tsplot":
            return self.tsplotw[icomp](*args)
        elif self.method == "roofit":
            return self.roofitw[icomp](*args)
        else:
            return sum(
                [
                    self.alphas[i, icomp] * self.pdfs[i](*args) / self.pdfnorms[i]
                    for i in range(self.ncomps)
                ]
            ) / sum(
                [
                    self.yields[i] * self.pdfs[i](*args) / self.pdfnorms[i]
                    for i in range(self.ncomps)
                ]
            )

    def make_weight_plot(
        self, axis=None, dopts=["r", "b", "g", "m", "c", "y"], labels=None
    ):
        """
        Make a plot of the weight functions.

        Parameters
        ----------
        axis : optional
            matplotlib axis to draw will default to use plt.gca()
        dopts : list of str
            List of colors for the different components
        labels : list of str
            List of legend labels. Default will use those passed in `compnames`
            with `__init__`
        """
        if self.ndiscvars != 1:
            print("WARNING - I dont know how to plot this")
            return None

        while len(dopts) < self.ncomps:
            dopts.append("b")

        try:
            import matplotlib.pyplot as plt
        except Exception:
            raise RuntimeError("matplotlib must be installed to make the weight plot")

        ax = axis or plt.gca()

        x = np.linspace(*self.discvarranges[0], 400)

        labels = labels or ["$w_{{{0}}}$".format(comp) for comp in self.compnames]

        for comp in range(self.ncomps):
            ax.plot(
                x,
                self.get_weight(comp, x),
                color=dopts[comp],
                linewidth=2,
                label=labels[comp],
            )

        label = labels[-1] if len(labels) > self.ncomps else r"$\sum_i w_{i}$"
        ax.plot(
            x,
            sum([self.get_weight(c, x) for c in range(self.ncomps)]),
            "k-",
            linewidth=3,
            label=label,
        )

        ax.legend()

    def print_checks(self):
        """Print checks."""
        if self.method != "roofit":
            self.intws = np.identity(self.ncomps)
            for i in range(self.ncomps):
                for j in range(self.ncomps):
                    self.intws[i, j] = nquad(
                        lambda *args: self.get_weight(i, *args)
                        * self.pdfs[j](*args)
                        / self.pdfnorms[j],
                        self.discvarranges,
                    )[0]
            print(
                """    Integral of w*pdf matrix (should be close to the
                identity):"""
            )
            # with np.printoptions(precision=3, suppress=True):
            print("\t" + str(self.intws).replace("\n", "\n\t"))

        print("    Check of weight sums (should match yields):")
        self.sows = [
            np.sum(self.get_weight(i, *self.data.T)) for i in range(self.ncomps)
        ]
        header = "\t{:10s} | {:^10s} | {:^10s} | {:^9s} |".format(
            "Component", "sWeightSum", "Yield", "Diff"
        )
        print(header)
        print("\t" + "-".join(["" for i in range(len(header) + 1)]))

        for i, yld in enumerate(self.yields):
            print(
                "\t  {:<8d} | {:10.4f} | {:10.4f} | {:8.2f}% |".format(
                    i, self.sows[i], yld, 100.0 * (yld - self.sows[i]) / self.sows[i]
                )
            )

    def _run_tsplot(self):

        import ROOT as r

        if self.method != "tsplot":
            raise RuntimeError(
                "Method is", self.method, "but calling the tsplot function."
            )

        # this works very differently for nD fits
        # so will not implement it here
        if self.ndiscvars != 1:
            raise RuntimeError("Sorry but I can't do the tsplot method for >1D fits")

        # make the tree for TSPlot

        # temporary datfile
        datfile = open(".data.dat", "w")
        for x in self.data:
            datfile.write("{}".format(x[0]))
            for i, pdf in enumerate(self.pdfs):
                x_pdf = pdf(x[0]) / self.pdfnorms[i]
                datfile.write(" {}".format(x_pdf))
            datfile.write("\n")
        datfile.close()

        # now make the tree and draw some plots to check it
        r.gROOT.SetBatch()

        tree = r.TTree("data", "data")
        read_str = "x/D:" + ":".join(["fx_%s" % a for a in self.compnames])

        tree.ReadFile(".data.dat", read_str, " ")
        c = r.TCanvas("c", "c", (self.ncomps + 1) * 600, 400)
        c.Divide(self.ncomps + 1, 1)
        c.cd(1)
        tree.Draw("x")
        for i in range(self.ncomps):
            c.cd(i + 2)
            tree.Draw("fx_%s:x" % self.compnames[i])
        c.Update()
        c.Modified()
        c.Draw()
        r.gErrorIgnoreLevel = r.kInfo
        c.Print("figs/tree.pdf")

        # now do the TSPlot
        from array import array

        tsplot = r.TSPlot(0, self.ndiscvars, len(self.data), self.ncomps, tree)
        sel_str = "x:" + ":".join(["fx_%s" % a for a in self.compnames])
        tsplot.SetTreeSelection(sel_str)
        ne = array("i", [int(yld) for yld in self.yields])
        tsplot.SetInitialNumbersOfSpecies(ne)
        tsplot.MakeSPlot("Q")

        # get the weights out of TSPlot
        weights = np.ndarray(len(self.data) * self.ncomps)
        tsplot.GetSWeights(weights)
        weights = np.reshape(weights, (-1, self.ncomps))
        data_w_weights = np.append(self.data, weights, axis=1)
        sorted_data_w_weights = data_w_weights[np.argsort(data_w_weights[:, 0])]

        return sorted_data_w_weights

    def _run_roofit(self):

        import ROOT as r

        if self.method != "roofit":
            raise RuntimeError(
                "Method is", self.method, "but calling the roofit function."
            )

        r.RooMsgService.instance().setGlobalKillBelow(r.RooFit.FATAL)
        r.gErrorIgnoreLevel = r.kInfo

        # sort out observables
        rf_obs = r.RooArgList()
        for obs in self.rfobs:
            if not obs.InheritsFrom("RooAbsReal"):
                raise RuntimeError(
                    """Found an observable which does not inherit from
                    RooAbsReal"""
                )
            rf_obs.add(obs)

        # make the roodataset and fill it
        rf_dset = r.RooDataSet("data", "data", r.RooArgSet(rf_obs))
        for row in self.data:
            for i in range(self.ndiscvars):
                rf_obs.at(i).setVal(row[i])
            rf_dset.add(r.RooArgSet(rf_obs))

        # pdfs should now be of the RooFit form
        rf_pdfs = r.RooArgList()
        for pdf in self.pdfs:
            if not pdf.InheritsFrom("RooAbsPdf"):
                raise RuntimeError("Found a pdf which does not inherit from RooAbsPdf.")
            # pdf.getParameters( rf_dset ).setAttribAll("Constant")
            rf_pdfs.add(pdf)

        # yields can still be numbers
        rfylds = [
            r.RooRealVar(
                "y_%s" % self.compnames[i],
                "y_%s" % self.compnames[i],
                yld,
                0.0,
                2.5 * yld,
            )
            for i, yld in enumerate(self.yields)
        ]
        rf_ylds = r.RooArgList()
        for yld in rfylds:
            rf_ylds.add(yld)

        # now can create the pdf
        rf_totpdf = r.RooAddPdf("pdf", "pdf", rf_pdfs, rf_ylds, False)

        # fit it (probably already been done)
        rf_totpdf.fitTo(rf_dset, r.RooFit.Extended(True), r.RooFit.PrintLevel(-1))

        # plot it
        c = r.TCanvas()
        for obs in self.rfobs:
            pl = obs.frame()
            rf_dset.plotOn(pl, r.RooFit.Binning(50))
            rf_totpdf.plotOn(pl)
            pl.Draw()
            c.Update()
            c.Modified()
            c.Draw()
            r.gErrorIgnoreLevel = r.kInfo
            c.Print("figs/rf_%s.pdf" % obs.GetName())

        # now get the sweights using RooStats
        r.RooStats.SPlot("sdata", "sdata", rf_dset, rf_totpdf, rf_ylds)

        weights = np.zeros((rf_dset.numEntries(), self.ncomps))

        for ev in range(rf_dset.numEntries()):
            rfvals = rf_dset.get(ev)
            for i, obs in enumerate(self.rfobs):
                assert abs(rfvals.getRealValue(obs.GetName()) - self.data[ev][i]) < 1e-6
            for i, comp in enumerate(self.compnames):
                weights[ev, i] = rfvals.getRealValue("y_" + comp + "_sw")

        data_w_weights = np.append(self.data, weights, axis=1)
        sorted_data_w_weights = data_w_weights[np.argsort(data_w_weights[:, 0])]

        return sorted_data_w_weights


def convert_rf_pdf(pdf, obs, npoints=400, forcenorm=False):
    """
    Convert RooAbsPdf into python callable.

    Helper function to convert a RooFit::RooAbsPdf object into a python
    callable that can be used by either the :class:`SWeight` or :class:`Cow`
    classes

    Parameters
    ----------
    pdf : RooAbsPdf
        The pdf, must inherit from RooAbsPdf (e.g. RooGaussian, RooExponential,
        RooAddPdf etc.)
    obs : RooRealVar
        The observable, must inherit from RooRealVarLValue but will usually be
        a RooRealVar
    npoints : int, optional
        The number of points to use for the interpolation
    forcenorm : bool, optional
        Force the return function to be normalised by performing a numerical
        integration of it (the function should in most cases be normalised
        properly anyway so this shouldn't be needed)

    Returns
    -------
    callable :
        A callable function representing a normalised pdf which can then be
        passed to the :class:`SWeight` or :class:`Cow` classes

    """
    try:
        import ROOT as r
    except Exception:
        raise RuntimeError("ROOT and RooFit must be installed to convert a RooAbsPdf")

    if not hasattr(obs, "InheritsFrom"):
        raise RuntimeError(
            """Observable does not appear to be a ROOT like object - it should
            inherit from RooAbsReal. Type: """,
            type(obs),
        )

    if not obs.InheritsFrom("RooAbsRealLValue"):
        raise RuntimeError(
            """Observable does not appear to be of the right type - it should
            inherit from RooAbsRealLValue. Type: """,
            type(obs),
        )

    range = (obs.getMin(), obs.getMax())

    xvals = np.linspace(*range, npoints)

    yvals = []

    normset = r.RooArgSet(obs)
    for x in xvals:
        obs.setVal(x)
        yvals.append(pdf.getVal(normset))

    f = InterpolatedUnivariateSpline(xvals, yvals)

    N = 1
    if forcenorm:
        N = nquad(f, (range,))[0]

    def retf(x):
        return f(x) / N

    return retf
