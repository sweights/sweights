# sweights

![](https://img.shields.io/pypi/v/sweights.svg)

```bash
pip install sweights
```

We provide a few tools for projecting component weights ("sweights") for a control variable(s) using a discriminating variable(s). For details please see [Dembinski, Kenzie, Langenbruch, Schmelling - arXiv:2112.04574](https://arxiv.org/abs/2112.04574).

We also provide tools for correcting the covariance matrix of fits to weighted data. For details please see [Dembinski, Kenzie, Langenbruch, Schmelling - arXiv:2112.04574](https://arxiv.org/abs/2112.04574) (Sec IV) and [Langenbruch - arXiv:1911.01303](https://arxiv.org/abs/1911.01303).

## Documentation

**The below documentation is now superseeded as it has been moved to readthedocs**. Please go to https://sweights.readthedocs.io for the latest.

A variety of options for extraction of signal-like weights is provided:

- classic "sWeights" via the `sweight` class produces pure weight functions and provides the Wkl and Akl matrices
  1. Using the "summation" method from the original sPlot paper [Pivk, Le Diberder - arXiv:physics/0402083](https://arxiv.org/abs/physics/0402083) referred to as Variant B in [arXiv:2112.04574](https://arxiv.org/abs/2112.04574)
  2. Using the "integration" method rederived originally by Schmelling and referred to as Variant A in [arXiv:2112.04574](https://arxiv.org/abs/2112.04574)
  3. Using the "refit" method, i.e. taking the covariance matrix of a yield only fit, referred to as Variant Ci in [arXiv:2112.04574](https://arxiv.org/abs/2112.04574)
  4. Using the "subhess" method, i.e. taking the sub-covariance matrix for the yields, referred to as Variant Cii in [arXiv:2112.04574](https://arxiv.org/abs/2112.04574)
  5. Using the implementation in ROOT's TSPlot (this we believe should be equivalent to Variant B but is more susceptible to numerical differences)
  6. Using the implementation in RooStat's SPlot (we found this identical to Variant B ("summation") above in all the cases we tried)

- Custom Orthogonal Weight functions (COWs) via the `cow` class produces pure weight functions and provides Wkl and Akl matrices. It expects
  - gs - the signal function for the discrimant variable
  - gb - the background function(s) for the discriminant variable (can pass orders of polynomials here if desired)
  - Im - the variance function for the discriminant variance (can also pass 1 and it will be set of uniform)
  - obs - one can instead or additionally pass the observed distribution in the discriminant variable which will be used for the variance function instead. In this case you must pass a two element tuple giving the bin entries and bin edges for the observed dataset (the same as what `np.histogram(data)` would return)

Corrections to the covariance matrix can be implemented using
- `cov_correct` which computes the full asymptotic correction using Eq. 55 in [arXiv:2112.04574](https://arxiv.org/abs/2112.04574)
- `approx_cov_correct` is more straightfoward to compute but only computes the first term in Eq. 55 of [arXiv:2112.04574](https://arxiv.org/abs/2112.04574) which will be slightly conservative

A test of variable independence based on the Kendall tau coefficient is also provided in `kendall_tau`

## Examples

An example script demonstrating a typical use case is provided in

```bash
python tests/examples.py
```

There is also a version written as a `.ipynb` in `doc/tutorial.ipynb`

This does the following:

1. **Generate toy data** in two independent dimensions *m* and *t*

  ![toy](https://user-images.githubusercontent.com/1140576/142237277-0485e6e7-8ccf-489a-affd-6b81028ed5c3.png)

2. **Fit the toy data** in the discriminanting variable to get an estimate of the discriminating variable pdfs

3. **Compute sWeights** using the "summation" method (implemented by the `sweight` class provided)

  ![sws](https://user-images.githubusercontent.com/1140576/142237391-0b37f428-5668-4602-98bb-097fdaae62e8.png)

4. **Compute sWeights** using the COW method with a variance function of unity, I(m)=1, (implemented by the `cow` class provided)

  ![cows](https://user-images.githubusercontent.com/1140576/142237453-8c3dfa2b-b38d-4e22-96d8-30f31f61d1c8.png)

5. **Fit the weighted distributions and correct the covariance** using the `cov_correct` function provided

  ![tfit](https://user-images.githubusercontent.com/1140576/142237505-11032b1c-b6fa-47dc-9a0e-e965210fdf6b.png)
