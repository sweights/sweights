.. _about:
.. _tutorial: notebooks/tutorial.ipynb
.. _converter: notebooks/roopdf_conversion.ipynb
.. toctree::
   :titlesonly:

About
=====

**sweights** is a package which can implement various different methods for projecting out a particular component in a control variable based on the distribution in a discriminating variable. It is based on the work described in `arXiv:2112.04574 <https://arxiv.org/abs/2112.04574>`_.

For a basic tour of what the package does then see the tutorial_.

If you are used to using `RooFit <https://root.cern/manual/roofit/>`_ and already have your fit setup this way then we also provide a wrapper to convert a RooFit pdf into the format expected here. You can take a look at the converter_.

The intention is to eventually port this package to `scikit-hep <https://scikit-hep.org>`_.

Maintainers
-----------

* Matthew Kenzie (@matthewkenzie)
* Hans Dembinski