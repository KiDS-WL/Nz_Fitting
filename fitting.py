import multiprocessing
import os
from functools import partial

import numpy as np
from scipy.optimize import curve_fit

from .data import DataTuple, BootstrapFit
from .models import BaseModel


class Optimizer(object):

    def __init__(self, data, model):
        assert(isinstance(data, DataTuple))
        self.data = data
        assert(isinstance(model, BaseModel))
        self.model = model

    def Ndof(self):
        """
        Degrees of freedom for model fit.
        """
        return len(self.data) - self.model.n_param

    def chisquare(self, bestfit):
        """
        chisquare(self, bestfit)

        Compute the fit chi^2 for a set of best-fit parameters.

        Parameters
        ----------
        bestfit : BootstrapFit
            Best-fit parameters used to evaluate the model.

        Returns
        -------
        chisq : BootstrapFit
        """
        model_n = self.model(self.data.z, *bestfit.paramBest())
        chisq = np.sum(((model_n - self.data.n) / self.data.dn)**2)
        return chisq

    def chisquareNdof(self, bestfit):
        """
        chisquare(self, bestfit)

        Compute the fit chi^2 per degrees of freedom for a set of best-fit
        parameters.

        Parameters
        ----------
        bestfit : BootstrapFit
            Best-fit parameters used to evaluate the model.

        Returns
        -------
        chisqNdof : BootstrapFit
        """
        chisqNdof = self.chisquare(bestfit) / self.Ndof()
        return chisqNdof

    def optimize(self, **kwargs):
        raise NotImplementedError


class CurveFit(Optimizer):
    """
    CurveFit(data, model)

    A wrapper for scipy.optmize.curve_fit to fit a comb model to a redshift
    distribution with a bootstrap estimate of the fit parameter covariance.

    Parameters
    ----------
    data : DataTuple
        Input redshift distribution.
    model : BaseModel
        Gaussian comb model with give number of components.
    """

    def __init__(self, data, model):
        super().__init__(data, model)

    def _curve_fit_wrapper(self, *args, resample=False, **kwargs):
        """
        _curve_fit_wrapper(resample=False)

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
            np.random.seed(  # reseed the RNG state for each thread
                int.from_bytes(os.urandom(4), byteorder='little'))
            fit_data = self.data.resample()
        else:
            fit_data = self.data
        popt, _ = curve_fit(
            self.model, fit_data.z, fit_data.n, sigma=fit_data.dn,
            p0=self.model.guess(), bounds=self.model.bounds(), **kwargs)
        return popt

    def optimize(self, n_samples=1000, threads=None, **kwargs):
        """
        optimize(resample=False, n_samples=1000, threads=-1, **kwargs)

        Computes the best fit parameters and their covariance from
        resampling and re-fitting the input data errors/covariance.

        Parameters
        ----------
        resample : bool
            Whether the data should be resampled.
        n_samples : int
            Number of resampling steps from data used to estimate the
            covariance.
        threads : int
            Number of threads used for processing (defaults to all threads).
        **kwargs : keyword arguments
            Arugments parsed on to scipy.optimize.curve_fit

        Returns
        -------
        bestfit : BootstrapFit
            Parameter best-fit container.
        """
        guess = self.model.guess()
        bounds = self.model.bounds()
        # get the best fit parameters
        pbest = self._curve_fit_wrapper()
        # resample data points for each fit to estimate parameter covariance
        if threads is None:
            threads = multiprocessing.cpu_count()
        threads = min(threads, n_samples)
        chunksize = n_samples // threads + 1  # optmizes the workload
        threaded_fit = partial(
            self._curve_fit_wrapper, resample=True, **kwargs)
        # run in parallel threads
        with multiprocessing.Pool(threads) as pool:
            param_samples = pool.map(
                threaded_fit, range(n_samples), chunksize=chunksize)
        bestfit = BootstrapFit(pbest, param_samples)
        return bestfit
