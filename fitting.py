import multiprocessing
import os
from collections import OrderedDict
from functools import partial

import numpy as np
from scipy.optimize import curve_fit

from .data import RedshiftData
from .models import BaseModel


class FitResult(object):
    """
    Container for best-fit parameters and and samples for correlatin
    estimation.

    Parameters
    ----------
    bestfit : OrderedDict
        Dictionary of with mapping parameter name -> best-fit model parameter.
    fitsamples : OrderedDict
        Dictionary of with mapping parameter name -> fit parameter samples.
    labels : OrderedDict
        Dictionary of with mapping parameter name -> math text / TEX label.
    n_dof : int
        Degrees of freedom of model fit.
    chisquare : float
        Chi squared of model best fit.
    """

    def __init__(self, bestfit, fitsamples, labels, n_dof, chisquare):
        assert(type(bestfit) is OrderedDict)
        self._best = bestfit
        assert(type(fitsamples) is OrderedDict)
        self._samples = fitsamples
        self._n_samples = len(self._samples.values().__iter__().__next__())
        assert(type(labels) is OrderedDict)
        self.names = labels
        self._ndof = n_dof
        self._chisquare = chisquare

    def __len__(self):
        return self._n_samples

    def __str__(self):
        max_width = max(len(n) for n in self.getParamNames())
        chi_str = "χ² dof."
        max_width = max(max_width, len(chi_str))
        string = "{:>{w}} = {:.3f}\n".format(
            chi_str, self.chiSquareReduced(), w=max_width)
        iterator = zip(
            self.getParamNames(), self.paramBest(), self.paramError())
        for i, (name, value, error) in enumerate(iterator):
            if i > 12:
                string += (" " * max_width) + "...     \n"
                break
            string += "{:>{w}} = {:}\n".format(
                name, format_variable(value, error, precision=3),
                w=max_width)
        return string.strip("\n")

    def __repr__(self):
        max_width = max(len(n) for n in self.getParamNames())
        string = "<%s object at %s,\n" % (
            self.__class__.__name__, hex(id(self)))
        for line in str(self).split("\n"):
            string += " %s\n" % line
        return string.strip("\n") + ">"

    def getParamNo(self):
        """
        Number of free model parameters.
        """
        return len(self._best)

    def getParamNames(self, label=False):
        """
        Names of the free model parameters.
        """
        if label:
            return self.names.values()
        else:
            return self.names.keys()

    def paramSamples(self, name=None):
        """
        Directly get the fit parameter samples.
        """
        if name is None:
            return np.transpose(list(self._samples.values()))
        else:
            return self._samples[name]

    def paramBest(self, name=None):
        """
        Get the best fit parameters.
        """
        if name is None:
            return np.asarray(list(self._best.values()))
        else:
            return self._best[name]

    def paramCovar(self):
        """
        Get the best fit parameter covariance matrix.
        """
        cov = np.cov(self.paramSamples(), rowvar=False)
        return np.atleast_2d(cov)

    def paramError(self, name=None):
        """
        Get the best fit parameter errors.
        """
        if name is None:
            return np.std(self.paramSamples(), axis=0)
        else:
            return np.std(self.paramSamples(name))

    def paramCorr(self):
        """
        Get the best fit parameter correlation matrix.
        """
        errors = self.paramError()
        covar = self.paramCovar()
        corr = np.matmul(
            np.matmul(np.diag(1.0 / errors), covar), np.diag(1.0 / errors))
        return corr

    def Ndof(self):
        return self._ndof

    def chiSquare(self):
        return self._chisquare
    
    def chiSquareReduced(self):
        return self.chiSquare() / self.Ndof()

    def paramAsTEX(
            self, param_name, precision=3, notation="auto", use_siunitx=False):
        """
        TODO
        """
        precision = max(0, precision)
        if param_name not in self.names:
            raise KeyError("parameter with name '%s' does not exist")
        # format to TEX, decide automatically which formatter to use
        expression = format_variable(
            self.paramBest(param_name), self.paramError(param_name),
            precision, TEX=True, notation=notation, use_siunitx=use_siunitx)
        TEXstring = "${:} = {:}$".format(
            self.names[param_name].strip("$"), expression.strip("$"))
        return TEXstring

    def plotSamples(self, names=None):
        """
        Plot the distribution of the fit parameter samples in a triangle plot
        with corner.corner.
        """
        if names is None:
            samples = self.paramSamples()
            labels = list(self.names.values())
        else:
            samples = []
            labels = []
            for name in names:
                samples.append(self.paramSamples(name))
                labels.append(self.names[name])
            samples = np.transpose(samples)
        fig = corner(samples, labels=labels, show_titles=True)
        return fig

    def plotCorr(self):
        """
        Plot the correlation matrix.
        """
        im = plt.matshow(self.paramCorr(), vmin=-1, vmax=1, cmap="bwr")
        plt.colorbar(im)


class Optimizer(object):

    def __init__(self, data, model):
        assert(isinstance(data, RedshiftData))
        self.data = data
        assert(isinstance(model, BaseModel))
        self.model = model

    def chiSquared(self, pbest):
        # compute the chi squared
        diff_data_model = self.model(self.data.z, *pbest) - self.data.n
        try:
            chisq = np.matmul(
                diff_data_model,
                np.matmul(self.model.getInverseCovariance(), diff_data_model))
        except AttributeError:
            chisq = np.sum((diff_data_model / self.data.dn)**2)
        return chisq

    def Ndof(self):
        """
        Degrees of freedom for model fit.
        """
        return len(self.data) - self.model.getParamNo()

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
        try:
            sigma = fit_data.getCovariance()
        except AttributeError:
            sigma = fit_data.dn
        # replace NaNs
        not_finite = ~np.isfinite(fit_data.n)
        if np.any(not_finite):
            fit_data.n[not_finite] = self.data.n[not_finite]
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
        label_dict = OrderedDict(zip(
            self.model.getParamNames(label=False),
            self.model.getParamNames(label=True)))
        guess = self.model.guess()
        bounds = self.model.bounds()
        # get the best fit parameters
        pbest = self._curve_fit_wrapper()
        pbest_dict = OrderedDict(zip(label_dict.keys(), pbest))
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
        param_samples_dict = OrderedDict(
            zip(label_dict.keys(), np.transpose(param_samples)))
        # collect the best fit data
        bestfit = FitParameters(
            pbest_dict, param_samples_dict, label_dict,
            self.Ndof(), self.chiSquared(pbest))
        return bestfit
