# Nz_Fitting
Repository for Nz fitting packages

## Features
- Redshift data structures with plotting routines
  - Data base class
  - Single redshift distribution
  - Set of tomographic bins
- Fitting models with plotting routines and mean redshift calculation
  - Model base class 
  - Simple bias model
  - Gaussian comb models with linear and logarithmic amplitude parameters
  - Multi-bin model a set of tomographic bins
- Fitting
  - Optimizer base class
  - Automatic wrapper for scipy.optimize.curve_fit

## Issues
- Currently does not include the bias model in the fitting procedure.
- Fitting the full KV450 sample yields a larger model uncertainty than the weighted sum of joint fits of all tomographic bins.

## Future Plans
- MCMC sampler (`emcee` or `PyMulitNest`) based on the optimizer base class
- Supporting joint n(z) and bias model fits
