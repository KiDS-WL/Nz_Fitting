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
  - Multi-bin models a set of tomographic bins
- Fitting
  - Optimizer base class
  - Automatic wrapper for scipy.optimize.curve_fit

## Issues
- Currently does not include the bias model in the fitting procedure.
- Fitting the full KV450 sample yields a larger model uncertainty than the weighted sum of joint fits of all tomographic bins.

## Future Plans
- MCMC sampler based on the Optimizer class
- Supporting joint n(z) and bias model fits
