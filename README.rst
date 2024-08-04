.. |sweights| image:: https://raw.githubusercontent.com/sweights/sweights/main/doc/_static/sweights_logo.svg
   :alt: sweights

|sweights|
==========

.. version-marker-do-not-remove

.. image:: https://img.shields.io/pypi/v/sweights.svg
  :target: https://pypi.org/project/sweights/
.. image:: https://github.com/sweights/sweights/actions/workflows/docs.yml/badge.svg?branch=main
  :target: https://sweights.github.io/sweights
.. image:: https://img.shields.io/badge/arXiv-2112.04574-b31b1b.svg
  :target: https://arxiv.org/abs/2112.04574

We provide a tool to calculate signal weights called *sWeights*, which can be used to project out the signal component in a mixture of signal and background in a control variable(s), while using fits in an independent discriminating variable. This technique was first popularized under the name *sPlot* method, but we think this is a misnomer and hence call it sWeights, since it is useful for more than plotting. We found that sWeights are a special case of more general Custom Orthogonal Weight functions (COWs), which extend the range of applicability of classic sWeights. If you use this package, please cite our paper:

`Dembinski, H., Kenzie, M., Langenbruch, C. and Schmelling, M., Custom Orthogonal Weight functions (COWs) for event classification, NIMA 1040 (2022) 167270 <https://www.sciencedirect.com/science/article/pii/S0168900222006076?via%3Dihub>`_

If you cannot access this paper for free, checkout the preprint `arXiv:2112.04574 <https://arxiv.org/abs/2112.04574>`_.

We also provide tools for computing the correct covariance matrix of fits to weighted data, described in section IV of our paper and in more detail in `Langenbruch arXiv:1911.01303 <https://arxiv.org/abs/1911.01303>`_. The standard method of inverting the Hesse matrix does not work. When in doubt, please use the bootstrap method.

Installation
------------

You can install sweights from PyPI.

.. code:: bash

    pip install sweights

Documentation
-------------

.. index-replace-marker-begin-do-not-remove

You can find `our documentation here <https://sweights.github.io/sweights>`_, which contain tutorials how to use the package and how avoid pitfalls.

.. index-replace-marker-end-do-not-remove

Partner projects
----------------

* `numba_stats`_ provides faster implementations of probability density functions than scipy, and a few specific ones used in particle physics that are not in scipy.
* `boost-histogram`_ from Scikit-HEP provides fast generalized histograms that you can use with the builtin cost functions.
* `jacobi`_ provides a robust, fast, and accurate calculation of the Jacobi matrix of any transformation function and building a function for generic error propagation.
* `resample`_ provides a simple API to calculate bootstrap estimate.

.. _numba_stats: https://github.com/HDembinski/numba-stats
.. _jacobi: https://github.com/HDembinski/jacobi
.. _boost-histogram: https://github.com/scikit-hep/boost-histogram
.. _resample: https://github.com/scikit-hep/resample
