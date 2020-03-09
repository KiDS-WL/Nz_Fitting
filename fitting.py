import multiprocessing
import warnings
from collections import OrderedDict
from functools import partial

import numpy as np
import pandas as pd
from corner import corner
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

from .data import Base
from .models import BaseModel
from .utils import format_variable


class FitResult(object):

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

    def nDoF(self):
        return self._ndof

    def chiSquare(self):
        return self._chisquare
    
    def chiSquareReduced(self):
        return self.chiSquare() / self.nDoF()

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

    def __init__(self, model, data):
        assert(isinstance(model, BaseModel))
        self._model = model
        assert(isinstance(data, Base))
        self._data = data

    def getData(self):
        return self._data

    def getModel(self):
        return self._model

    def chiSquared(self, params, ndof=False, use_cov=True):
        model = self._model._optimizerCall(self._data, *params)  # concatenated
        data = self._data.n(concat=True)
        diff_data_model = model - data
        if self._data.hasCovMat() and use_cov:
            invmat = self._data.getCovMat(inv=True)
            chisq = np.matmul(
                diff_data_model, np.matmul(invmat, diff_data_model))
        else:
            errors = self._data.dn(concat=True)
            chisq = np.sum((diff_data_model / errors)**2)
        if ndof:
            chisq /= self.nDoF()
        return chisq

    def nDoF(self):
        n_data = self._data.len(all=False, concat=True)
        n_param = self._model.getParamNo()
        return n_data - n_param


class CurveFit(Optimizer):

    def __init__(self, model, data):
        super().__init__(model, data)

    def _curve_fit_wrapper(
            self, *args, draw_sample=False, **kwargs):
        # get the data sample to fit
        if draw_sample:
            fit_data = self._data.getSample(args[0])  # the sample index
        else:
            fit_data = self._data  # fiducial fit
        # get the covariance matrix if possible
        if fit_data.hasCovMat() and fit_data.getCovMatType() != "diagonal":
            sigma = fit_data.getCovMat()
        else:
            sigma = fit_data.dn(concat=True)
        # run the optimizer
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(
                self._model._optimizerCall,
                fit_data, fit_data.n(concat=True), sigma=sigma,
                p0=self._model.getParamGuess(),
                bounds=self._model.getParamBounds(pairwise=False),
                **kwargs)
        return popt

    def optimize(self, n_samples=None, threads=None, **kwargs):
        label_dict = OrderedDict(zip(
            self._model.getParamNames(),
            self._model.getParamlabels()))
        guess = self._model.getParamGuess()
        bounds = self._model.getParamBounds()
        # get the best fit parameters
        pbest = self._curve_fit_wrapper(draw_sample=False, **kwargs)
        self._model.setParamGuess(pbest)
        pbest_dict = OrderedDict(zip(label_dict.keys(), pbest))
        # resample data points for each fit to estimate parameter covariance
        if threads is None:
            threads = multiprocessing.cpu_count()
        # get the number of samples to use
        if self._data.hasSamples():
            if n_samples is None:
                n_samples = self._data.getSampleNo()
            else:
                n_samples = min(n_samples, self._data.getSampleNo())
        else:
            n_samples = 1000  # default value
        threads = min(threads, n_samples)
        chunksize = n_samples // threads + 1  # optmizes the workload
        threaded_fit = partial(
            self._curve_fit_wrapper, draw_sample=True, **kwargs)
        # run in parallel threads
        with multiprocessing.Pool(threads) as pool:
            param_samples = pool.map(
                threaded_fit, range(n_samples), chunksize=chunksize)
        param_samples_dict = OrderedDict(
            zip(label_dict.keys(), np.transpose(param_samples)))
        # collect the best fit data
        bestfit = FitResult(
            pbest_dict, param_samples_dict, label_dict,
            self.nDoF(), self.chiSquared(pbest))
        return bestfit
