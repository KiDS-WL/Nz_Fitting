import numpy as np
from matplotlib import pyplot as plt

from .data import FitParameters

sqrt_two_pi = np.sqrt(2.0 * np.pi)


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
    prefactor = np.full_like(x, 1.0 / (sqrt_two_pi * sigma))
    exponent = -0.5 * ((x - mu) / sigma)**2
    return prefactor * np.exp(exponent)


class BaseModel(object):

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
        return (-np.inf, np.inf)

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

    def modelError(self, bestfit, z, percentile=68.0):
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
        n_z_samples = np.empty((bestfit.n_samples, len(z)))
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

    def meanError(self, bestfit, z):
        """
        Determine the standard error on the mean of the redshfit model for
        given best fit parameters.

        Parameters
        ----------
        bestfit : FitParameters
            Best-fit parameters used to evaluate the model.
        z : array_like
            Redshifts at with the model is sampled.

        Returns
        -------
        z_mean_err : float
            Standard error of the Means of redshift model.
        """
        # compute the mean redshift for each fit parameter sample
        z_mean_samples = np.empty(bestfit.n_samples)
        for i, params in enumerate(bestfit.paramSamples()):
            z_mean_samples[i] = self.mean(params, z)
        z_mean_err = z_mean_samples.std()
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
            ax = plt.gca()
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


class PowerLawBias(BaseModel):
    """
    Simple bias model of the form (1 + z)^a.
    """

    n_param = 1

    def guess(self):
        return np.zeros(self.n_param)

    def bounds(self):
        return (np.full(self.n_param, -5.0), np.full(self.n_param, 5.0))

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


class CombModel(BaseModel):

    def __init__(self, n_param, z0, dz, smoothing=1.0):
        assert(smoothing >= 1.0)
        # distribute the components
        self.z0 = z0
        self.dz = dz
        self.n_param = n_param
        self.mus = np.arange(z0, z0 + n_param * dz, dz)
        # set the width / overlap between the components
        self.smoothing = smoothing
        self.sigmas = np.full_like(self.mus, dz * smoothing)

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

    def modelError(self, bestfit, z=None, percentile=68.0):
        if z is None:
            z = self.autoSampling()
        z, n_z_min, n_z_max = super().modelError(bestfit, z, percentile)
        return z, n_z_min, n_z_max

    def mean(self, bestfit, z=None):
        if z is None:
            z = self.autoSampling()
        z_mean = super().mean(bestfit, z)
        return z_mean

    def meanError(self, bestfit, z=None):
        if z is None:
            z = self.autoSampling()
        z_mean_err = super().meanError(bestfit, z)
        return z_mean_err

    def plot(self, bestfit, z=None, ax=None, **kwargs):
        if z is None:
            z = self.autoSampling()
        super().plot(bestfit, z, ax, **kwargs)


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
        return np.ones(self.n_param)

    def bounds(self):
        return (np.full(self.n_param, 0.0), np.full(self.n_param, np.inf))

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
        return np.zeros(self.n_param)

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


class BinnedRedshiftModel(BaseModel):

    def __init__(self, bins, weights):
        assert(all(isinstance(m, BaseModel) for m in bins))
        assert(len(weights) == len(bins))
        self._models = [m for m in bins]
        self.weights = np.asarray(weights) / sum(weights)
        self.n_models = len(bins)
        self.n_param_model = [m.n_param for m in bins]
        self.n_param = sum(self.n_param_model)

    def guess(self):
        # concatenate the guesess from the bin models
        guess = np.concatenate([m.guess() for m in self._models])
        return guess

    def bounds(self):
        # concatenate the bounds from the bin models
        lower_bound = np.concatenate([m.bounds()[0] for m in self._models])
        upper_bound = np.concatenate([m.bounds()[1] for m in self._models])
        return (lower_bound, upper_bound)

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
        # split the parameters
        bin_params = np.split(params, np.cumsum(self.n_param_model)[:-1])
        # split the redshifts assuming there is the same number per sample
        bin_z = np.split(z, self.n_models + 1)
        # evaluate each model
        bin_n_z = []
        master_n_z = np.zeros_like(bin_z[-1])
        for i in range(self.n_models):
            n_z = self._models[i](bin_z[i], *bin_params[i])
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

    def modelError(self, bestfit, z=None, percentile=68.0):
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
        n_z_samples = np.empty((bestfit.n_samples, len(z), len(z[0])))
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
        z_mean : list
            List of mean of redshift model by bin/master sample.
        """
        if z is None:
            z = self.autoSampling()
        binned_z, binned_n_z = self.modelBest(bestfit, z)
        z_means = np.array([
            np.average(bz, weights=bn_z)
            for bz, bn_z in zip(binned_z, binned_n_z)])
        return z_means

    def meanError(self, bestfit, z=None):
        """
        Determine the standard error on the mean of the redshfit model for
        given best fit parameters for each bin and the master sample.

        Parameters
        ----------
        bestfit : FitParameters
            Best-fit parameters used to evaluate the model.
        z : list of array_like
            A list of sampling points, split by bin/master sample at with the
            model is sampled.

        Returns
        -------
        z_mean_err : list
            List of standard error of the Means of redshift model by bin/master
            sample.
        """
        if z is None:
            z = self.autoSampling()
        # compute the mean redshift for each fit parameter sample
        z_mean_samples = np.empty((bestfit.n_samples, self.n_models + 1))
        for i, params in enumerate(bestfit.paramSamples()):
            z_mean_samples[i] = self.mean(params, z)
        z_means_err = z_mean_samples.std(axis=0)
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
            n_data = self.n_models + 1
            # try to arrange the subplots in a grid
            n_x = int(np.ceil(n_data / np.sqrt(n_data)))
            n_y = int(np.ceil(n_data / n_x))
            fig, axes = plt.subplots(
                n_y, n_x, figsize=(3 * n_x, 3 * n_y), sharex=True, sharey=True)
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
