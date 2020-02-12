from copy import copy
from math import pi, sqrt

import numpy as np
from matplotlib import pyplot as plt
from scipy.special import erfinv
from scipy.interpolate import interp1d

from .data import RedshiftHistogram, RedshiftData, FitParameters
from .utils import Figure


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


class BaseModel(object):

    def getParamNo(self):
        """
        Number of free model parameters.
        """
        raise NotImplementedError

    def getParamNames(self, label=False):
        """
        Names of the free model parameters.
        """
        raise NotImplementedError

    def guess(self):
        """
        Parameter guess for the amplitudes in the format required by
        scipy.optmize.curve_fit.
        """
        raise NotImplementedError

    def bounds(self):
        """
        Parameter bounds for the amplitudes in the format required by
        scipy.optmize.curve_fit.
        """
        raise NotImplementedError

    def __call__(self):
        """
        Factory method for the model implementation.
        """
        raise NotImplementedError

    def autoSampling(self):
        """
        Factory method to automatically sample the model.
        """
        raise NotImplementedError

    def modelBest(self, bestfit, z):
        """
        Evaluate the fit model with the best fit parameters.

        Parameters
        ----------
        bestfit : FitParameters or array_like
            Best-fit parameters used to evaluate the model.
        z : array_like
            Redshifts at with the model is sampled.

        Returns
        -------
        z : array_like
            Redshifts at with the model is sampled.
        n_z : array_like
            Model value.
        """
        try:
            n_z = self(z, *bestfit.paramBest())
        except AttributeError:
            n_z = self(z, *bestfit)
        return z, n_z

    def modelError(self, bestfit, z, percentile=68.3):
        """
        Determine the fit upper and lower constraint given a percentile.

        Parameters
        ----------
        bestfit : FitParameters
            Best-fit parameters used to evaluate the model.
        z : array_like
            Redshifts at with the model is sampled.
        percentile : float
            Percentile to use for the model constraint, must be between 0.0
            and 100.0.

        Returns
        -------
        z : array_like
            Redshifts at with the model is sampled.
        n_z_min : array_like
            Lower model constraint.
        n_z_max : array_like
            Upper model constraint.
        """
        # evaluate the model for each fit parameter sample
        n_z_samples = np.empty((len(bestfit), len(z)))
        for i, params in enumerate(bestfit.paramSamples()):
            _, n_z_samples[i] = self.modelBest(params, z)
        # compute the range of the percentiles of the model distribution
        p = (100.0 - percentile) / 2.0
        n_z_min, n_z_max = np.percentile(n_z_samples, [p, 100.0 - p], axis=0)
        return z, n_z_min, n_z_max

    def mean(self, bestfit, z):
        """
        Determine the mean of the redshfit model for given best fit parameters.

        Parameters
        ----------
        bestfit : FitParameters or array_like
            Best-fit parameters used to evaluate the model.
        z : array_like
            Redshifts at with the model is sampled.

        Returns
        -------
        z_mean : float
            Mean of redshift model.
        """
        z, n_z = self.modelBest(bestfit, z)
        z_mean = np.average(z, weights=n_z)
        return z_mean

    def meanError(self, bestfit, z, percentile=68.3, symmetric=True):
        """
        Determine the uncertainty on the mean of the redshfit model for given
        best fit parameters. Returns the standard error by default.

        Parameters
        ----------
        bestfit : FitParameters
            Best-fit parameters used to evaluate the model.
        z : array_like
            Redshifts at with the model is sampled.
        percentile : float
            Percentile to use for the model constraint, must be between 0.0
            and 100.0.
        symmetric : bool
            Whether the upper and lower constraints should be symmetric.

        Returns
        -------
        z_mean_err : list of float
            Lower and upper constraint on the uncertainty.
        """
        # compute the mean redshift for each fit parameter sample
        z_mean_samples = np.empty(len(bestfit))
        for i, params in enumerate(bestfit.paramSamples()):
            z_mean_samples[i] = self.mean(params, z)
        if symmetric:
            z_mean_err = z_mean_samples.std()
            # scale sigma to match the requested percentile
            nsigma = np.sqrt(2.0) * erfinv(percentile / 100.0)
            z_mean_err *= nsigma
            z_mean_err = [-z_mean_err, z_mean_err]
        else:
            p = (100.0 - percentile) / 2.0
            z_mean_min, z_mean_max = np.percentile(
                z_mean_samples, [p, 100.0 - p])
            z_mean = self.mean(bestfit, z)
            z_mean_err = [z_mean_min - z_mean, z_mean_max - z_mean]
        return z_mean_err

    def plot(self, bestfit, z, ax=None, **kwargs):
        """
        Plot the model and its uncertainty based on a set of best fit
        parameters.

        Parameters
        ----------
        bestfit : FitParameters
            Best-fit parameters used to evaluate the model.
        z : array_like
            Redshifts at with the model is sampled.
        ax : matplotlib.axes
            Specifies the axis to plot on.
        **kwargs : keyword arguments
            Arugments parsed on to matplotlib.pyplot.plot and fill_between

        Returns
        -------
        bestfit : FitParameters
            Parameter best-fit container.
        """
        if ax is None:
            fig = Figure(1)
            ax = plt.gca()
        else:
            fig = plt.gcf()
        plot_kwargs = {}
        plot_kwargs.update(kwargs)
        line = ax.plot(*self.modelBest(bestfit, z), **plot_kwargs)[0]
        # add a shaded area that indicates the 68% model confidence
        try:
            plot_kwargs.pop("color")
        except KeyError:
            pass
        ax.fill_between(
            *self.modelError(bestfit, z), alpha=0.3,
            color=line.get_color(), **plot_kwargs)
        return fig


class PowerLawBias(BaseModel):
    """
    Simple bias model of the form (1 + z)^α.
    """

    def getParamNo(self):
        return 1

    def getParamNames(self, label=False):
        if label:
            return tuple([r"$\alpha$"])
        else:
            return tuple(["alpha"])

    def guess(self):
        return np.zeros(self.getParamNo())

    def bounds(self):
        n_param = self.getParamNo()
        return (np.full(n_param, -5.0), np.full(n_param, 5.0))

    def __call__(self, z, *params):
        """
        Evaluate the model on a redshift sampling with given bias
        parameters. Works with scipy.optmize.curve_fit.

        Parameters
        ----------
        z : array_like
            Points at with the model is evaluated.
        *params : float
            Set of bias parameters.

        Returns
        -------
        b_z : array_like
            Model evaluated at z.
        """
        b_z = (1.0 + z) ** params[0]
        return b_z


class BiasFitModel(BaseModel):
    """
    Compute a model for a redshift measurement from a sum of redshift bins and
    a given model for the galaxy bias. The model free parameters are the
    parameters of the bias model.

    TODO: MISSING!
    """

    def __init__(self, bias, bins, full, weights):
        assert(isinstance(bias, BaseModel))
        self._bias = bias
        assert(all(isinstance(tb, RedshiftData) for tb in bins))
        self.bins = bins
        assert(isinstance(full, RedshiftData))
        self.full = full
        assert(len(bins) == len(weights))
        self.weights = np.asarray(weights) / sum(weights)

    def getParamNo(self):
        return self._bias.getParamNo()

    def getParamNames(self, label=False):
        return self._bias.getParamNames(label)

    def guess(self):
        return self._bias.guess()

    def bounds(self):
        return self._bias.bounds()

    def __call__(self, z, *params):
        """
        Compute a model from the sum of tomographic bins that

        Parameters
        ----------
        z : array_like
            Dummy variable, automatically take from master sample.
        *params : float
            Set of bias parameters.

        Returns
        -------
        b_z : array_like
            Model evaluated at z.
        """
        bias = self._bias(self.full.z, *params)  # evaluate the given bias model
        # evaluate the renormalisation of the full sample
        renorm = np.trapz(self.full.n / bias, x=self.full.z)
        # apply the bias correction to the bins and renormalize them as well
        bin_nz = []
        for nz in self.bins:
            nz_debiased = nz.n / bias
            nz_debiased /= np.trapz(nz_debiased, x=nz.z)
            bin_nz.append(nz_debiased)
        # compute the weighted sum of the bias corrected bins
        nz_sum = np.sum([
            nz * w for w, nz in zip(self.weights, bin_nz)], axis=0)
        nz_full_model = bias * renorm * nz_sum
        return nz_full_model

    def plot(self, bestfit, ax=None, **kwargs):
        return super().plot(bestfit, self.full.z, ax, **kwargs)


class CombModel(BaseModel):

    def __init__(self, n_param, z0, dz, smoothing=1.0):
        assert(smoothing >= 1.0)
        # distribute the components
        self.z0 = z0
        self.dz = dz
        self._n_param = n_param
        self.mus = np.arange(z0, z0 + n_param * dz, dz)
        # set the width / overlap between the components
        self.smoothing = smoothing
        self.sigmas = np.full_like(self.mus, dz * smoothing)

    def getParamNo(self):
        """
        Number of free model parameters.
        """
        return copy(self._n_param)

    def getParamNames(self, label=False):
        """
        Names of the free model parameters.
        """
        if label:
            return tuple(
                "$A_{%d}$" % (i + 1) for i in range(self.getParamNo()))
        else:
            return tuple("A_%d" % (i + 1) for i in range(self.getParamNo()))

    def autoSampling(self, n=200):
        """
        Automatic sampling of the model based on the spread of the components.

        Parameters
        ----------
        n : int
            Number of sampling points to generate.

        Returns
        -------
        z : array_like
            Sampling points.
        """
        z = np.linspace(0.0, self.mus[-1] + 3.0 * self.sigmas[-1], n)
        return z

    def modelBest(self, bestfit, z=None):
        if z is None:
            z = self.autoSampling()
        z, n_z = super().modelBest(bestfit, z)
        return z, n_z

    def modelError(self, bestfit, z=None, percentile=68.3):
        if z is None:
            z = self.autoSampling()
        z, n_z_min, n_z_max = super().modelError(bestfit, z, percentile)
        return z, n_z_min, n_z_max

    def mean(self, bestfit, z=None):
        if z is None:
            z = self.autoSampling()
        z_mean = super().mean(bestfit, z)
        return z_mean

    def meanError(self, bestfit, z=None, percentile=68.3, symmetric=True):
        if z is None:
            z = self.autoSampling()
        z_mean_err = super().meanError(bestfit, z, percentile, symmetric)
        return z_mean_err

    def plot(self, bestfit, z=None, ax=None, **kwargs):
        if z is None:
            z = self.autoSampling()
        return super().plot(bestfit, z, ax, **kwargs)


class GaussianComb(CombModel):
    """
    Redshift comb model with Gaussian components (distributed uniformly along
    the redshift axis) multiplied by redshift. The free parameters are the
    (linear) amplitudes of the components. The stanard deviation of these
    Gaussians is at least equal to their distance and can be increased with a
    smoothing factor.

    Parameters
    ----------
    n_param : int
        Number of Gaussian components.
    x0 : float
        Redshift of the first component.
    dn : float
        Distance in redshift between the components.
    smoothing : float
        Widening factor of the component standard distribution, must be >= 1.
    """

    def __init__(self, n_param, z0, dz, smoothing=1.0):
        super().__init__(n_param, z0, dz, smoothing)

    def guess(self):
        return np.ones(self.getParamNo())

    def bounds(self):
        n_param = self.getParamNo()
        return (np.full(n_param, 0.0), np.full(n_param, np.inf))

    def __call__(self, z, *params):
        """
        Evaluate the model on a redshift sampling with given amplitude
        parameters. Works with scipy.optmize.curve_fit.

        Parameters
        ----------
        z : array_like
            Points at with the model is evaluated.
        *params : float
            Set of linear component amplitudes.

        Returns
        -------
        n_z : array_like
            Model evaluated at z.
        """
        p_iter = zip(params, self.mus, self.sigmas)
        n_z = z * np.sum([
            amp * gaussian(z, mu, sig) for amp, mu, sig in p_iter], axis=0)
        return n_z


class LogGaussianComb(CombModel):
    """
    Redshift comb model with Gaussian components (distributed uniformly along
    the redshift axis) multiplied by redshift. The free parameters are the
    (logarithmic) amplitudes of the components. The stanard deviation of these
    Gaussians is at least equal to their distance and can be increased with a
    smoothing factor.

    Parameters
    ----------
    n_param : int
        Number of Gaussian components.
    x0 : float
        Redshift of the first component.
    dn : float
        Distance in redshift between the components.
    smoothing : float
        Widening factor of the component standard distribution, must be >= 1.
    """

    def __init__(self, n_param, z0, dz, smoothing=1.0):
        super().__init__(n_param, z0, dz, smoothing)

    def guess(self):
        return np.zeros(self.getParamNo())

    def __call__(self, z, *params):
        """
        Evaluate the model on a redshift sampling with given amplitude
        parameters. Works with scipy.optmize.curve_fit.

        Parameters
        ----------
        z : array_like
            Points at with the model is evaluated.
        *params : float
            Set of logarithmic component amplitudes.

        Returns
        -------
        n_z : array_like
            Model evaluated at z.
        """
        p_iter = zip(params, self.mus, self.sigmas)
        n_z = z * np.sum([
            np.exp(amp) * gaussian(z, mu, sig) for amp, mu, sig in p_iter],
            axis=0)
        return n_z


class ShiftModel(BaseModel):

    def __init__(self, histogram, data_binning):
        self._data_binning = np.asarray(data_binning)
        assert(isinstance(histogram, RedshiftHistogram))
        self._hist = histogram
        # compute the probability density
        self._cdf = interp1d(
            self._hist.z, self._hist.cdf, fill_value="extrapolate")

    def getParamNo(self):
        """
        Number of free model parameters.
        """
        return 2

    def getParamNames(self, label=False):
        """
        Names of the free model parameters.
        """
        if label:
            return (r"$\delta z$", r"$A$")
        else:
            return ("dz", "A")

    def guess(self):
        """
        Parameter guess for the amplitudes in the format required by
        scipy.optmize.curve_fit.
        """
        return np.array([0.0, 1.0])

    def bounds(self):
        """
        Parameter bounds for the amplitudes in the format required by
        scipy.optmize.curve_fit.
        """
        return (np.array([-0.5, 0.0]), np.array([0.5, np.inf]))

    def __call__(self, z, *params):
        # If the input redshift sampling doesn't match the one store interally,
        # reconstruct it approximately based on z. This is necessary to get a
        # correct shift fit, but at he same time being able to plot at an
        # arbitrary redshift resolution.
        if len(z) + 1 != len(self._data_binning):
            shifted_bins = np.append(
                0.0, np.append((z[1:] + z[:-1]) / 2.0, z.max() + 0.1))
        else:
            shifted_bins = self._data_binning + params[0]
        P_edges = self._cdf(shifted_bins)
        pdf_shifted = np.diff(P_edges) / np.diff(shifted_bins)
        # renormalize
        norm = np.trapz(pdf_shifted, x=z)
        return pdf_shifted / norm * params[1]


class ShiftModelBinned(BaseModel):

    def __init__(self, shiftmodels, bias=None):
        assert(all(type(m) is ShiftModel for m in shiftmodels))
        self._models = [m for m in shiftmodels]
        self.n_models = len(shiftmodels)
        self.n_param_models = [m.getParamNo() for m in shiftmodels]
        if bias is not None:
            assert(isinstance(bias, BaseModel))
        self._bias = bias

    def getParamNo(self):
        n_param = sum(m.getParamNo() for m in self._models)
        if self._bias is not None:
            n_param += self._bias.getParamNo()
        return n_param

    def getParamNames(self, label=False):
        names = []
        if label:
            for i, model in enumerate(self._models, 1):
                names.extend(
                    pn[:-1] + "_{%d}$" % i
                    for pn in model.getParamNames(label))
        else:
            for i, model in enumerate(self._models, 1):
                names.extend(
                    pn + "_%d" % i for pn in model.getParamNames())
        if self._bias is not None:
            names.extend(self._bias.getParamNames(label))
        return tuple(names)

    def guess(self):
        # concatenate the guesses from the bin models
        guess = [m.guess() for m in self._models]
        if self._bias is not None:
            guess.append(self._bias.guess())
        return np.concatenate(guess)

    def bounds(self):
        # concatenate the bounds from the bin models
        lower_bound = [m.bounds()[0] for m in self._models]
        upper_bound = [m.bounds()[1] for m in self._models]
        if self._bias is not None:
            lower_bound.append(self._bias.bounds()[0])
            upper_bound.append(self._bias.bounds()[1])
        return (np.concatenate(lower_bound), np.concatenate(upper_bound))

    def __call__(self, z, *params):
        """
        TODO

        Parameters
        ----------
        z : array_like
            Points at with the model is evaluated. These must be a
            concatenation of the sampling points of all bins and the master
            sample.
        *params : float
            Concatenation of the set of logarithmic component amplitudes for
            the bins.

        Returns
        -------
        n_z : array_like
            Model evaluated for each bin and the master sample concatenated to
            a single data vector.
        """
        # split parameters, last array is empty or holds bias model params
        param_tuples = np.split(params, np.cumsum(self.n_param_models))
        # split the redshifts assuming there is the same number per sample
        bin_z = np.split(z, self.n_models)
        # evaluate each model
        bin_n_z = []
        for i in range(self.n_models):
            n_z = self._models[i](bin_z[i], *param_tuples[i])
            if self._bias is not None:  # multiply with bias to mach data
                bias_params = param_tuples[-1]
                n_z *= self._bias(bin_z[i], *bias_params)
            bin_n_z.append(n_z)
        n_z = np.concatenate(bin_n_z)
        return n_z

    def modelBest(self, bestfit, z):
        """
        Evaluate the fit model with the best fit parameters, splitting the
        data by bins/master sample.

        Parameters
        ----------
        bestfit : FitParameters or array_like
            Best-fit parameters used to evaluate the model.
        z : list of array_like
            A list of sampling points, split by bin/master sample at with the
            model is sampled.

        Returns
        -------
        z : list of array_like
            List of sampling points for each bin and the master sample.
        n_z : list of array_like
            List of model values for each bin and the master sample.
        """
        try:
            n_z = self(np.concatenate(z), *bestfit.paramBest())
        except AttributeError:
            n_z = self(np.concatenate(z), *bestfit)
        n_z = np.split(n_z, self.n_models)
        return z, n_z

    def modelError(self, bestfit, z, percentile=68.3):
        """
        Determine the fit upper and lower constraint given a percentile,
        splitting the data by bins/master sample.

        Parameters
        ----------
        bestfit : FitParameters or array_like
            Best-fit parameters used to evaluate the model.
        z : list of array_like
            A list of sampling points, split by bin/master sample at with the
            model is sampled.
        percentile : float
            Percentile to use for the model constraint, must be between 0.0
            and 100.0.

        Returns
        -------
        z : list of array_like
            List of sampling points for each bin and the master sample.
        n_z_min : list of array_like
            List of model lower constraints for each bin and the master sample.
        n_z_max : list of array_like
            List of model upper constraints for each bin and the master sample.
        """
        n_z_samples = np.empty((len(bestfit), len(z), len(z[0])))
        for i, params in enumerate(bestfit.paramSamples()):
            _, n_z_samples[i] = self.modelBest(params, z)
        p = (100.0 - percentile) / 2.0
        n_z_min, n_z_max = np.percentile(
            n_z_samples, [p, 100.0 - p], axis=0)
        return z, [n for n in n_z_min], [n for n in n_z_max]

    def mean(self, bestfit):
        """
        Determine the mean of the redshfit model for given best fit parameters
        for each bin and the master sample.

        Parameters
        ----------
        bestfit : FitParameters or array_like
            Best-fit parameters used to evaluate the model.
        z : list of array_like
            A list of sampling points, split by bin/master sample at with the
            model is sampled.

        Returns
        -------
        z_mean : array_like
            List of mean of redshift model by bin/master sample.
        """
        binned_z, binned_n_z = self.modelBest(bestfit, z)
        z_means = np.array([
            np.average(bz, weights=bn_z)
            for bz, bn_z in zip(binned_z, binned_n_z)])
        return z_means

    def meanError(self, bestfit, percentile=68.3, symmetric=True):
        """
        Determine the uncertainty on the mean of the redshfit model for given
        best fit parameters for each bin and the master sample. Returns the
        standard errors by default.

        Parameters
        ----------
        bestfit : FitParameters
            Best-fit parameters used to evaluate the model.
        z : list of array_like
            A list of sampling points, split by bin/master sample at with the
            model is sampled.
        percentile : float
            Percentile to use for the model constraint, must be between 0.0
            and 100.0.
        symmetric : bool
            Whether the upper and lower constraints should be symmetric.

        Returns
        -------
        z_means_err : array_like
            List of pairs of lower and upper constraints on the undertainty of
            the means of the redshift model by bin/master sample.
        """
        # compute the mean redshift for each fit parameter sample
        z_means_samples = np.empty((len(bestfit), self.n_models))
        for i, params in enumerate(bestfit.paramSamples()):
            z_means_samples[i] = self.mean(params, z)
        if symmetric:
            z_means_err = z_means_samples.std(axis=0)
            # scale sigma to match the requested percentile
            nsigma = np.sqrt(2.0) * erfinv(percentile / 100.0)
            z_means_err *= nsigma
            z_means_err = np.transpose([-z_means_err, z_means_err])
        else:
            p = (100.0 - percentile) / 2.0
            z_means_min, z_means_max = np.percentile(
                z_means_samples, [p, 100.0 - p], axis=0)
            z_means = self.mean(bestfit, z)
            z_means_err = np.transpose([
                z_means_min - z_means, z_means_max - z_means])
        return z_means_err

    def plot(self, bestfit, z, fig=None, **kwargs):
        """
        Plot the model and its uncertainty based on a set of best fit
        parameters. Tomographic bins are arranged in a grid of separate plots
        followed by the (full) master sample.

        Parameters
        ----------
        bestfit : FitParameters
            Best-fit parameter used to evaluate the model.
        z : list of array_like
            A list of sampling points, split by bin/master sample at with the
            model is sampled.
        fig : matplotlib.figure
            Plot on an existig figure which must have at least n_data axes.
        **kwargs : keyword arguments
            Arugments parsed on to matplotlib.pyplot.plot and fill_between

        Returns
        -------
        fig : matplotlib.figure
            The figure containting the plots.
        """
        if fig is None:
            fig = Figure(self.n_data)
            axes = np.asarray(fig.axes)
        else:
            axes = np.asarray(fig.axes)
        plot_kwargs = {}
        plot_kwargs.update(kwargs)
        # compute the plot values
        binned_z, binned_n_z = self.modelBest(bestfit, z)
        binned_z, binned_n_z_min, binned_n_z_max = self.modelError(bestfit, z)
        for i, ax in enumerate(axes.flatten()):
            line = ax.plot(binned_z[i], binned_n_z[i], **plot_kwargs)[0]
            # add a shaded area that indicates the 68% model confidence
            try:
                plot_kwargs.pop("color")
            except KeyError:
                pass
            ax.fill_between(
                binned_z[i], binned_n_z_min[i], binned_n_z_max[i], alpha=0.3,
                color=line.get_color(), **plot_kwargs)
        return fig


class CombModelBinned(BaseModel):
    """
    TODO: MISSING!
    """

    def __init__(self, bins, weights, bias=None):
        assert(all(isinstance(m, CombModel) for m in bins))
        assert(len(weights) == len(bins))
        self._models = [m for m in bins]
        self.weights = np.asarray(weights) / sum(weights)
        self.n_models = len(bins)
        self.n_param_models = [m.getParamNo() for m in bins]
        if bias is not None:
            assert(isinstance(bias, BaseModel))
        self._bias = bias

    def getParamNo(self):
        n_param = sum(self.n_param_models)
        if self._bias is not None:
            n_param += self._bias.getParamNo()
        return n_param

    def getParamNames(self, label=False):
        names = []
        for i, model in enumerate(self._models):
            if label:
                names.extend(
                    [r"$A_{%d,%d}$" % (i + 1, j + 1)
                    for j in range(model.getParamNo())])
            else:
                names.extend(
                    ["A_%d,%d" % (i + 1, j + 1)
                    for j in range(model.getParamNo())])
        if self._bias is not None:
            names.extend(self._bias.getParamNames(label))
        return tuple(names)

    def guess(self):
        # concatenate the guesses from the bin models
        guess = [m.guess() for m in self._models]
        if self._bias is not None:
            guess.append(self._bias.guess())
        return np.concatenate(guess)

    def bounds(self):
        # concatenate the bounds from the bin models
        lower_bound = [m.bounds()[0] for m in self._models]
        upper_bound = [m.bounds()[1] for m in self._models]
        if self._bias is not None:
            lower_bound.append(self._bias.bounds()[0])
            upper_bound.append(self._bias.bounds()[1])
        return (np.concatenate(lower_bound), np.concatenate(upper_bound))

    def __call__(self, z, *params):
        """
        Evaluate the model on a redshift sampling with given amplitude
        parameters. Evaluates the bin models and computes their weighted sum as
        model for the master sample. Works with scipy.optmize.curve_fit.

        Parameters
        ----------
        z : array_like
            Points at with the model is evaluated. These must be a
            concatenation of the sampling points of all bins and the master
            sample.
        *params : float
            Concatenation of the set of logarithmic component amplitudes for
            the bins.

        Returns
        -------
        n_z : array_like
            Model evaluated for each bin and the master sample concatenated to
            a single data vector.
        """
        # split parameters, last array is empty or holds bias model params
        param_tuples = np.split(params, np.cumsum(self.n_param_models))
        # split the redshifts assuming there is the same number per sample
        bin_z = np.split(z, self.n_models + 1)
        # evaluate each model
        bin_n_z = []
        master_n_z = np.zeros_like(bin_z[-1])
        for i in range(self.n_models):
            n_z = self._models[i](bin_z[i], *param_tuples[i])
            if self._bias is not None:  # multiply with bias to mach data
                bias_params = param_tuples[-1]
                n_z *= self._bias(bin_z[i], *bias_params)
            # compute the wighted sum of bins which models the master sample
            master_n_z += n_z * self.weights[i]
            bin_n_z.append(n_z)
        n_z = np.concatenate([*bin_n_z, master_n_z])
        return n_z

    def autoSampling(self, n=200):
        """
        Automatic sampling of the model based on the spread of the components.
        Invokes the autoSampling method of each bin and generates, based on
        these, a sampling for the master sample.

        Parameters
        ----------
        n : int
            Number of sampling points to generate per bin.

        Returns
        -------
        z : list of array_like
            List of sampling points for each bin and the master sample.
        """
        z = [m.autoSampling(n) for m in self._models]
        z.append(
            np.linspace(0.0, max(bz.max() for bz in z), n))
        return z

    def modelBest(self, bestfit, z=None):
        """
        Evaluate the fit model with the best fit parameters, splitting the
        data by bins/master sample.

        Parameters
        ----------
        bestfit : FitParameters or array_like
            Best-fit parameters used to evaluate the model.
        z : list of array_like
            A list of sampling points, split by bin/master sample at with the
            model is sampled.

        Returns
        -------
        z : list of array_like
            List of sampling points for each bin and the master sample.
        n_z : list of array_like
            List of model values for each bin and the master sample.
        """
        if z is None:
            z = self.autoSampling()
        try:
            n_z = self(np.concatenate(z), *bestfit.paramBest())
        except AttributeError:
            n_z = self(np.concatenate(z), *bestfit)
        n_z = np.split(n_z, self.n_models + 1)
        return z, n_z

    def modelError(self, bestfit, z=None, percentile=68.3):
        """
        Determine the fit upper and lower constraint given a percentile,
        splitting the data by bins/master sample.

        Parameters
        ----------
        bestfit : FitParameters or array_like
            Best-fit parameters used to evaluate the model.
        z : list of array_like
            A list of sampling points, split by bin/master sample at with the
            model is sampled.
        percentile : float
            Percentile to use for the model constraint, must be between 0.0
            and 100.0.

        Returns
        -------
        z : list of array_like
            List of sampling points for each bin and the master sample.
        n_z_min : list of array_like
            List of model lower constraints for each bin and the master sample.
        n_z_max : list of array_like
            List of model upper constraints for each bin and the master sample.
        """
        if z is None:
            z = self.autoSampling()
        n_z_samples = np.empty((len(bestfit), len(z), len(z[0])))
        for i, params in enumerate(bestfit.paramSamples()):
            _, n_z_samples[i] = self.modelBest(params, z)
        p = (100.0 - percentile) / 2.0
        n_z_min, n_z_max = np.percentile(
            n_z_samples, [p, 100.0 - p], axis=0)
        return z, [n for n in n_z_min], [n for n in n_z_max]

    def mean(self, bestfit, z=None):
        """
        Determine the mean of the redshfit model for given best fit parameters
        for each bin and the master sample.

        Parameters
        ----------
        bestfit : FitParameters or array_like
            Best-fit parameters used to evaluate the model.
        z : list of array_like
            A list of sampling points, split by bin/master sample at with the
            model is sampled.

        Returns
        -------
        z_mean : array_like
            List of mean of redshift model by bin/master sample.
        """
        if z is None:
            z = self.autoSampling()
        binned_z, binned_n_z = self.modelBest(bestfit, z)
        z_means = np.array([
            np.average(bz, weights=bn_z)
            for bz, bn_z in zip(binned_z, binned_n_z)])
        return z_means

    def meanError(self, bestfit, z=None, percentile=68.3, symmetric=True):
        """
        Determine the uncertainty on the mean of the redshfit model for given
        best fit parameters for each bin and the master sample. Returns the
        standard errors by default.

        Parameters
        ----------
        bestfit : FitParameters
            Best-fit parameters used to evaluate the model.
        z : list of array_like
            A list of sampling points, split by bin/master sample at with the
            model is sampled.
        percentile : float
            Percentile to use for the model constraint, must be between 0.0
            and 100.0.
        symmetric : bool
            Whether the upper and lower constraints should be symmetric.

        Returns
        -------
        z_means_err : array_like
            List of pairs of lower and upper constraints on the undertainty of
            the means of the redshift model by bin/master sample.
        """
        if z is None:
            z = self.autoSampling()
        # compute the mean redshift for each fit parameter sample
        z_means_samples = np.empty((len(bestfit), self.n_models + 1))
        for i, params in enumerate(bestfit.paramSamples()):
            z_means_samples[i] = self.mean(params, z)
        if symmetric:
            z_means_err = z_means_samples.std(axis=0)
            # scale sigma to match the requested percentile
            nsigma = np.sqrt(2.0) * erfinv(percentile / 100.0)
            z_means_err *= nsigma
            z_means_err = np.transpose([-z_means_err, z_means_err])
        else:
            p = (100.0 - percentile) / 2.0
            z_means_min, z_means_max = np.percentile(
                z_means_samples, [p, 100.0 - p], axis=0)
            z_means = self.mean(bestfit, z)
            z_means_err = np.transpose([
                z_means_min - z_means, z_means_max - z_means])
        return z_means_err

    def plot(self, bestfit, z=None, fig=None, **kwargs):
        """
        Plot the model and its uncertainty based on a set of best fit
        parameters. Tomographic bins are arranged in a grid of separate plots
        followed by the (full) master sample.

        Parameters
        ----------
        bestfit : FitParameters
            Best-fit parameter used to evaluate the model.
        z : list of array_like
            A list of sampling points, split by bin/master sample at with the
            model is sampled.
        fig : matplotlib.figure
            Plot on an existig figure which must have at least n_data axes.
        **kwargs : keyword arguments
            Arugments parsed on to matplotlib.pyplot.plot and fill_between

        Returns
        -------
        fig : matplotlib.figure
            The figure containting the plots.
        """
        if fig is None:
            fig = Figure(self.n_data)
            axes = np.asarray(fig.axes)
        else:
            axes = np.asarray(fig.axes)
        if z is None:
            z = self.autoSampling()
        plot_kwargs = {}
        plot_kwargs.update(kwargs)
        # compute the plot values
        binned_z, binned_n_z = self.modelBest(bestfit, z)
        binned_z, binned_n_z_min, binned_n_z_max = self.modelError(bestfit, z)
        for i, ax in enumerate(axes.flatten()):
            line = ax.plot(binned_z[i], binned_n_z[i], **plot_kwargs)[0]
            # add a shaded area that indicates the 68% model confidence
            try:
                plot_kwargs.pop("color")
            except KeyError:
                pass
            ax.fill_between(
                binned_z[i], binned_n_z_min[i], binned_n_z_max[i], alpha=0.3,
                color=line.get_color(), **plot_kwargs)
        return fig
