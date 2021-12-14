import ROOT as r
from ROOT import RooFit as rf

mass = r.RooRealVar('m','m',5000,5800)

# gaussian
mean = r.RooRealVar('mu','mu',5350,5300,5400)
sigma = r.RooRealVar('sg','sg',25,0,100)
spdf = r.RooGaussian('sig','sig',mass,mean,sigma)

# exponential
slope = r.RooRealVar('lb','lb',-0.002,-0.01,0)
bpdf = r.RooExponential('bkg','bkg',mass,slope)

# total
sy = r.RooRealVar('sy','sy',1000,0,10000)
by = r.RooRealVar('by','by',2000,0,10000)
pdf = r.RooAddPdf('pdf','pdf',r.RooArgList(spdf,bpdf), r.RooArgList(sy,by))

# plot RooFit style
c = r.TCanvas()
pl = mass.frame(rf.Bins(800))
pdf.plotOn(pl, rf.Normalization(1,r.RooAbsReal.NumEvent), rf.Precision(-1), rf.Components("bkg"), rf.LineColor(r.kRed), rf.LineStyle(r.kDashed) )
pdf.plotOn(pl, rf.Normalization(1,r.RooAbsReal.NumEvent), rf.Precision(-1), rf.Components("sig"), rf.LineColor(r.kGreen+3) )
pdf.plotOn(pl, rf.Normalization(1,r.RooAbsReal.NumEvent), rf.Precision(-1) )
pl.Draw()
c.Update()


# now do the conversion
# this should return every pdf normalised
from sweights import convertRooAbsPdf
_spdf = convertRooAbsPdf(spdf,mass)
_bpdf = convertRooAbsPdf(bpdf,mass)
_pdf = convertRooAbsPdf(pdf,mass)

# now draw the function mpl style
import matplotlib.pyplot as plt
import numpy as np
fs = sy.getVal()/(sy.getVal()+by.getVal())
fb = by.getVal()/(sy.getVal()+by.getVal())

x = np.linspace(mass.getMin(), mass.getMax(), 400)
plt.plot(x, fs*_spdf(x), 'g-')
plt.plot(x, fb*_bpdf(x), 'r--')
plt.plot(x, _pdf(x), 'b-')
plt.show()



