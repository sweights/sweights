.. sweights documentation master file

Documentation for the sweights package
======================================

**These docs are for sweights version:** |release|

.. toctree::
    :hidden:

    install
    reference
    tutorials
    contribute

.. _converter: notebooks/roopdf_conversion.ipynb
.. _RooFit: https://root.cern/manual/roofit
.. _Scikit-HEP: https://scikit-hep.org

.. toctree::
   :titlesonly:

**sweights** is a package which can implement various different methods for projecting out a particular component in a control variable based on the distribution in a discriminating variable. It is based on the work described in `H. Dembinski, M. Kenzie, C. Langenbruch, M. Schmelling, NIMA 1040 (2022) 167270 <https://arxiv.org/abs/2112.04574>`_.

For a basic tour of what the package does have a look at the :ref:`tutorials` and at the  :ref:`reference`. Alternatively you can browse the :ref:`genindex`.

If you are using RooFit_ to fit the component PDFs, you can still use this package after converting them, see converter_.

The intention is to eventually port this package to Scikit-HEP_.

Maintainers
-----------

* Matthew Kenzie (@matthewkenzie)
* Hans Dembinski
