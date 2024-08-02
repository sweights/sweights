.. _reference:

API Reference
=============

.. currentmodule:: sweights

Quick Summary
-------------
Here are the main classes and methods:

.. autosummary::
   Cows
   SWeight
   Cow
   convert_rf_pdf
   kendall_tau
   cov_correct
   approx_cov_correct

COWs (experimental)
-------------------

This is a new API for sWeights computation which will replace the current API when we switch to v2.0.0. The API in v1.x is experimental and may change from version to version. It will become stable from v2.x onward. If you need a stable API right now, please use the SWeight and Cow classes below or pin the version of the sweights package.

.. autoclass:: Cows
   :members:
   :special-members: __call__, __getitem__, __len__

Classic sWeights
----------------

.. autoclass:: SWeight
   :members:

COWs
----

.. autoclass:: Cow
   :members:

Covariance Correction
---------------------

.. autofunction:: cov_correct
.. autofunction:: approx_cov_correct

Utilities
---------

.. automodule:: sweights.util
   :members:

.. currentmodule:: sweights
.. autofunction:: kendall_tau
.. autofunction:: convert_rf_pdf
