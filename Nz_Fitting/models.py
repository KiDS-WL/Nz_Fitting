from copy import copy
from math import pi, sqrt

import numpy as np
from matplotlib import pyplot as plt
from scipy.special import erfinv
from scipy.interpolate import interp1d

from .data import RedshiftData, RedshiftDataBinned
from .utils import Figure


class FitParameter:

    guess = 0
    lower = -np.inf
    upper = np.inf

    def __init__(self, name, label):
        self.name = name
        self.label = label

    def __copy__(self):
        new = FitParameter(self.name, self.label)
        new.setGuess(self.guess)
        new.setBounds(self.lower, self.upper)
        return new

    def __str__(self):
        string = "parameter '{:}' ∈ {:}, {:}".format(
            self.name,
            "(-∞" if np.isneginf(self.lower) else ("[" + str(self.lower)),
            "∞)" if np.isposinf(self.upper) else (str(self.upper) + "]"))
        return string

    def setGuess(self, guess):
        self.guess = guess

    def setBounds(self, lower=-np.inf, upper=np.inf):
        self.lower = lower
        self.upper = upper


class BaseModel(object):

    _paramlist = None
    _expect_z = "z"

    def getParams(self):
        return self._paramlist

    def getParamNo(self):
        return len(self._paramlist)

    def getParamNames(self):
        return [p.name for p in self._paramlist]

    def getParamlabels(self):
        return [p.label for p in self._paramlist]

    def getParamGuess(self):
        return [p.guess for p in self._paramlist]

    def setParamGuess(self, guesses):
        assert(len(guesses) == len(self._paramlist))
        for param, guess in zip(self._paramlist, guesses):
            param.setGuess(guess)

    def getParamBounds(self, pairwise=False):
        if pairwise:
            return [(p.lower, p.upper) for p in self._paramlist]
        else:
            lower = [p.lower for p in self._paramlist]
            upper = [p.upper for p in self._paramlist]
            return lower, upper

    @staticmethod
    def _parseWeights(weights):
        weights = np.array(weights)
        # check weights
        master = weights[-1]
        sum_bins = weights[:-1].sum()
        if not np.isclose(sum_bins, master):
            raise ValueError("weights of bins must sum up to master sample")
        weights /= sum_bins
        return weights

    def getZ(self, data):
        try:
            return getattr(data, self._expect_z)(all=False, concat=False)
        except AttributeError:
            return data

    def _optimizerCall(self, data, *params):
        try:
            return np.concatenate(self(data, *params))
        except ValueError:
            return self(data, *params)

    def _evalZ(self, z_data):
        # get the data needed to evaluate the model and make the plot
        if hasattr(z_data, "centers"):
            z = z_data.centers()
        elif hasattr(z_data, "z"):
            z = z_data.z(all=(self._expect_z == "edges"))
        else:
            z = copy(z_data)
        if self._expect_z == "edges":
            if hasattr(z_data, "edges"):
                z_model = z_data.edges()
            else:
                # the input are bin edges so we must compute the centers
                z_model = copy(z_data)
                try:
                    z = (z_model[1:] + z_model[:-1]) / 2.0
                except TypeError:
                    z = [(edges[1:] + edges[:-1]) / 2.0 for edges in z_model]
        else:
            z_model = z
        return z, z_model

    def evaluate(self, z_data, params):
        z, z_model = self._evalZ(z_data)
        # call the model
        try:
            param_samples = params.paramSamples()
            n = self(z_model, *params.paramBest())
            samples = np.empty((len(param_samples), len(n)))
            for i, param in enumerate(param_samples):
                samples[i] = self(z_model, *param)
            dn = samples.std(axis=0)
        except AttributeError:
            n = self(z_model, *params)
            samples = None
            dn = np.zeros_like(n)
        # pack the data as a RedshiftData container
        container = RedshiftData(z, n, dn)
        if samples is not None:
            container.setSamples(samples)
        return container


class BaseModelBinned(BaseModel):

    def _getFig(self, fig):
        if fig is None:
            try:
                n_plots = len(self)
            except TypeError:
                n_plots = 1
            fig = Figure(n_plots)
            axes = np.asarray(fig.axes)
        else:
            axes = np.asarray(fig.axes)
        return fig, axes

    def evaluate(self, z_data, params):
        z, z_model = self._evalZ(z_data)
        # call the model
        try:
            param_samples = params.paramSamples()
            n = self(z_model, *params.paramBest())
            samples = [
                np.empty((len(param_samples), len(bin_n))) for bin_n in n]
            for i, param in enumerate(param_samples):
                for j, bin_sample in enumerate(self(z_model, *param)):
                    samples[j][i] = bin_sample
            dn = [bin_sample.std(axis=0) for bin_sample in samples]
        except AttributeError:
            n = self(z_model, *params)
            samples = None
            dn = [np.zeros_like(bin_n) for bin_n in n]
        # pack the data as a RedshiftDataBinned container
        bins = []
        for i in range(len(n)):
            container = RedshiftData(z[i], n[i], dn[i])
            if samples is not None:
                container.setSamples(samples[i])
            bins.append(container)
        container = RedshiftDataBinned(bins[:-1], bins[-1])
        return container


class PowerLawBias(BaseModel):

    def __init__(self):
        param = FitParameter("alpha", r"$\alpha$")
        param.setGuess(0.0)
        param.setBounds(lower=-5.0, upper=5.0)
        self._paramlist = [param]

    def __call__(self, z_data, *params):
        z = self.getZ(z_data)
        return (1.0 + z) ** params[0]


class ShiftModel(BaseModel):

    _expect_z = "edges"

    def __init__(self, hist):
        self._model_hist = hist
        # create the parameter list
        shift_param = FitParameter("dz", r"$\delta z$")
        shift_param.setGuess(0.0)
        shift_param.setBounds(lower=-1.0, upper=1.0)
        amp_param = FitParameter("A", r"$A$")
        amp_param.setGuess(1.0)
        amp_param.setBounds(lower=0.0)
        self._paramlist = [shift_param, amp_param]

    def __call__(self, z_data, *params):
        z = self.getZ(z_data)
        try:
            edges_shifted = z - params[0]
        except TypeError:
            edges_shifted = z.edges() - params[0]
        # compute the CDF from a shifted (different) binning
        cdf_shifted = self._model_hist.cdf(edges_shifted)
        cdf_shifted /= cdf_shifted[-1]  # normalize
        # compute the new PDF
        pdf_shifted = np.diff(cdf_shifted) / np.diff(edges_shifted)
        # mask out model values where the data is not defined
        if hasattr(z_data, "mask"):
            pdf_shifted = pdf_shifted[~z_data.mask()]
        return pdf_shifted * params[1]  # rescale to the data


class ShiftModelBinned(BaseModelBinned):

    _expect_z = "edges"

    def __init__(self, shift_models, bias_model=None):
        self._models = shift_models
        self._bias_model = bias_model
        # create the parameter list
        self._param_per_model = []
        self._paramlist = []
        for i, model in enumerate(shift_models, 1):
            for param in model.getParams():
                new = copy(param)
                # modify the name
                new.name = param.name + "_{:d}".format(i)
                # modify the label
                new.label = "${:}_{{{:d}}}$".format(
                    param.label.strip("$"), i)
                self._paramlist.append(new)
            self._param_per_model.append(model.getParamNo())
        if self._bias_model is not None:
            self._paramlist.extend(bias_model.getParams())
            self._param_per_model.append(bias_model.getParamNo())

    def __call__(self, z_data, *params):
        assert(len(z_data) == len(self._models))
        z = self.getZ(z_data)
        # split parameters, last array is empty or holds bias model params
        param_tuples = np.split(params, np.cumsum(self._param_per_model))
        # evaluate each model
        bin_pdfs = []
        for i, bin_z in enumerate(z):
            pdf = self._models[i](bin_z, *param_tuples[i])
            if self._bias_model is not None:  # multiply with bias to mach data
                pdf *= self._bias(bin_z, *param_tuples[-1])
            bin_pdfs.append(pdf)
        # mask out model values where the data is not defined
        if hasattr(z_data, "getData"):
            for i, data in enumerate(z_data.getData()):
                bin_pdfs[i] = bin_pdfs[i][~data.mask()]
        return bin_pdfs


class CombModel(BaseModel):

    def __init__(self, n_param, z0, dz, smoothing=1.0):
        self._mus, self._sigmas = self._distributeGaussians(
            n_param, z0, dz, smoothing)
        # generate the parameter list
        self._paramlist = []
        for i in range(1, n_param + 1):
            self._paramlist.append(
                FitParameter("A_{:d}".format(i), r"$A_{{{:d}}}$".format(i)))

    @staticmethod
    def _distributeGaussians(n, z0, dz, smoothing):
        # distribute the components
        assert(smoothing >= 1.0)
        mus = np.arange(z0, z0 + n * dz, dz)
        sigmas = np.full_like(mus, dz * smoothing)
        return mus, sigmas

    def _checkParamNo(self, params):
        if len(params) != len(self._paramlist):
            raise ValueError(
                "expected {:d} parameters but got {:d}".format(
                    len(self._paramlist), len(params)))

    @staticmethod
    def gaussian(x, mu, sigma):
        """
        Sample a normal distribution function with mean and standard deviation.

        Parameters
        ----------
        x : array_like
            Points at with the distribution is sampled.
        mu : float
            Mean of the distribution.
        sigma : float
            Standard deviation of the distribution.

        Returns
        -------
        pos_sphere : array_like
            Normal distribution sampled at x.
        """
        prefactor = np.full_like(x, 1.0 / (sqrt(2.0  *pi) * sigma))
        exponent = -0.5 * ((x - mu) / sigma)**2
        return prefactor * np.exp(exponent)


class GaussianComb(CombModel):

    def __init__(self, n_param, z0, dz, smoothing=1.0):
        super().__init__(n_param, z0, dz, smoothing)
        for param in self._paramlist:
            param.setGuess(1.0)
            param.setBounds(lower=0.0)

    def __call__(self, z_data, *params):
        self._checkParamNo(params)
        z = self.getZ(z_data)
        if len(params) != len(self._paramlist):
            raise ValueError(
                "expected {:d} parameters but got {:d}".format(
                    len(self._paramlist), len(params)))
        n_z = z * np.sum([
            amp * self.gaussian(z, mu, sig)
            for amp, mu, sig in
            zip(params, self._mus, self._sigmas)], axis=0)
        return n_z


class LogGaussianComb(CombModel):

    def __init__(self, n_param, z0, dz, smoothing=1.0):
        super().__init__(n_param, z0, dz, smoothing)
        for param in self._paramlist:
            param.setGuess(0.0)

    def __call__(self, z_data, *params):
        self._checkParamNo(params)
        z = self.getZ(z_data)
        n_z = z * np.sum([
            np.exp(amp) * self.gaussian(z, mu, sig)
            for amp, mu, sig in
            zip(params, self._mus, self._sigmas)], axis=0)
        return n_z


class CombModelBinned(BaseModelBinned):

    def __init__(self, comb_models, weights, bias_model=None):
        self._models = comb_models
        self._bias_model = bias_model
        self._weights = self._parseWeights(weights)
        # create the parameter list
        self._param_per_model = []
        self._paramlist = []
        for i, model in enumerate(comb_models, 1):
            for param in model.getParams():
                new = copy(param)
                # modify the name
                name, suffix = param.name.split("_")
                new.name = "_{:d},".format(i).join([name, suffix])
                # modify the label
                label, suffix = param.label.split("{")
                new.label = "{{{:d},".format(i).join([label, suffix])
                self._paramlist.append(new)
            self._param_per_model.append(model.getParamNo())
        if self._bias_model is not None:
            self._paramlist.extend(bias_model.getParams())
            self._param_per_model.append(bias_model.getParamNo())

    def __call__(self, z_data, *params):
        if len(z_data) != (len(self._models) + 1):
            raise ValueError(
                "expected {:d} redshift samplings for {:} bins".format(
                    len(self._models) + 1, len(self._models)))
        z = self.getZ(z_data)
        # split parameters, last array is empty or holds bias model params
        param_tuples = np.split(params, np.cumsum(self._param_per_model))
        # evaluate each model
        bin_nzs = []
        master_nz = np.zeros(len(z[-1]))
        for i, bin_z in enumerate(z[:-1]):
            nz = self._models[i](bin_z, *param_tuples[i])
            if self._bias_model is not None:  # multiply with bias to mach data
                nz *= self._bias(bin_z, *param_tuples[-1])
            # compute the wighted sum of bins which models the master sample
            master_nz += nz * self._weights[i]
            bin_nzs.append(nz)
        return [*bin_nzs, master_nz]
