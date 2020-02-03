import multiprocessing
import os
from functools import partial

import numpy as np
from scipy.optimize import curve_fit

from .data import FitParameters, RedshiftData
from .models import BaseModel


class Optimizer(object):

    def __init__(self, data, model):
        assert(isinstance(data, RedshiftData))
        self.data = data
        assert(isinstance(model, BaseModel))
        self.model = model

    def Ndof(self):
        """
        Degrees of freedom for model fit.
        """
        return len(self.data) - self.model.getParamNo()

    def chisquare(self, bestfit):
        """
        Compute the chi^2 for a set of best-fit parameters.

        Parameters
        ----------
        bestfit : FitParameters
            Best-fit parameters used to evaluate the model.

        Returns
        -------
        chisq : float
        """
        model_n = self.model(self.data.z, *bestfit.paramBest())
        chisq = np.sum(((model_n - self.data.n) / self.data.dn)**2)
        return chisq

    def chisquareReduced(self, bestfit):
        """
        Compute the reduced chi^2 for a set of best-fit parameters.

        Parameters
        ----------
        bestfit : FitParameters
            Best-fit parameters used to evaluate the model.

        Returns
        -------
        chisq_red : float
        """
        chisq_red = self.chisquare(bestfit) / self.Ndof()
        return chisq_red

    def optimize(self, **kwargs):
        raise NotImplementedError


class CurveFit(Optimizer):
    """
    A wrapper for scipy.optmize.curve_fit to fit a comb model to a redshift
    distribution with a bootstrap estimate of the fit parameter covariance.
    Automatically includes the data covariance if provided in the input data
    container.

    Parameters
    ----------
    data : RedshiftData
        Input redshift distribution.
    model : BaseModel
        Gaussian comb model with give number of components.
    """

    def __init__(self, data, model):
        super().__init__(data, model)

    def _curve_fit_wrapper(self, *args, resample=False, **kwargs):
        """
        Internal wrapper for scipy.optimize.curve_fit to automatically parse
        fit data and model. The data can be resampled prior to fitting.

        Parameters
        ----------
        *args : objects
            Placeholder for dummy variables.
        resample : bool
            Whether the data should be resampled.
        **kwargs : keyword arguments
            Arugments parsed on to scipy.optimize.curve_fit

        Returns
        -------
        popt : array_like
            Best fit model parameters.
        """
        if resample:
            if self.data.reals is not None:
                fit_data = self.data.resample(reals_idx=args[0])
            else:
                np.random.seed(  # reseed the RNG state for each thread
                    int.from_bytes(os.urandom(4), byteorder="little"))
                fit_data = self.data.resample()
        else:
            fit_data = self.data
        # get the covariance matrix if possible
        if fit_data.cov is not None:
            sigma = fit_data.cov
        else:
            sigma = fit_data.dn
        # run the optimizer
        popt, _ = curve_fit(
            self.model, fit_data.z, fit_data.n, sigma=sigma,
            p0=self.model.guess(), bounds=self.model.bounds(), **kwargs)
        return popt

    def optimize(self, n_samples=1000, threads=None, **kwargs):
        """
        Computes the best fit parameters and their covariance from either
        data vector realisations or resampling data errors/covariance.

        Parameters
        ----------
        resample : bool
            Whether the data should be resampled.
        n_samples : int
            Number of resampling steps from data used to estimate the
            covariance (no effect if input data have realisations).
        threads : int
            Number of threads used for processing (defaults to all threads).
        **kwargs : keyword arguments
            Arugments parsed on to scipy.optimize.curve_fit

        Returns
        -------
        bestfit : FitParameters
            Parameter best-fit container.
        """
        guess = self.model.guess()
        bounds = self.model.bounds()
        # get the best fit parameters
        pbest = self._curve_fit_wrapper()
        # resample data points for each fit to estimate parameter covariance
        if threads is None:
            threads = multiprocessing.cpu_count()
        if self.data.reals is not None:
            n_samples = self.data.getNoRealisations()
        threads = min(threads, n_samples)
        chunksize = n_samples // threads + 1  # optmizes the workload
        threaded_fit = partial(
            self._curve_fit_wrapper, resample=True, **kwargs)
        # run in parallel threads
        with multiprocessing.Pool(threads) as pool:
            param_samples = pool.map(
                threaded_fit, range(n_samples), chunksize=chunksize)
        bestfit = FitParameters(pbest, param_samples, self.model)
        return bestfit
