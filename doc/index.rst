.. sweights documentation master file

Documentation for the sweights package
======================================

**These docs are for sweights version:** |release|

.. toctree::
    :hidden:

    install
    reference
    notebooks/tutorial
    notebooks/factorization_test
    notebooks/roopdf_conversion
    contribute

.. _about:
.. _tutorial: notebooks/tutorial.ipynb
.. _converter: notebooks/roopdf_conversion.ipynb
.. toctree::
   :titlesonly:

**sweights** is a package which can implement various different methods for projecting out a particular component in a control variable based on the distribution in a discriminating variable. It is based on the work described in `H. Dembinski, M. Kenzie, C. Langenbruch, M. Schmelling, NIMA 1040 (2022) 167270 <https://arxiv.org/abs/2112.04574>`_.

For a basic tour of what the package does then see the tutorial_ and have a look at the  :ref:`api`. Alternatively you can browse the :ref:`genindex`.

If you are used to using `RooFit <https://root.cern/manual/roofit/>`_ and already have your fit setup this way then we also provide a wrapper to convert a RooFit pdf into the format expected here, see converter_.

The intention is to eventually port this package to `Scikit-HEP <https://scikit-hep.org>`_.

Maintainers
-----------

* Matthew Kenzie (@matthewkenzie)
* Hans Dembinski