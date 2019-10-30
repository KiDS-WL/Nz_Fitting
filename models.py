import numpy as np
from matplotlib import pyplot as plt

sqrt_two_pi = np.sqrt(2.0 * np.pi)


def gaussian(x, mu, sigma):
    """
    gaussian(x, mu, sigma)

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
        modelBest(bestfit, z)

        Evaluate the fit model with the best fit parameters.

        Parameters
        ----------
        bestfit : BootstrapFit
            Best-fit parameter used to evaluate the model.
        z : array_like
            Redshifts at with the model is sampled.

        Returns
        -------
        z : array_like
            Redshifts at with the model is sampled.
        n_z : array_like
            Model value.
        """
        n_z = self(z, *bestfit.paramBest())
        return z, n_z

    def modelError(self, bestfit, z, percentile=68.0):
        """
        modelError(z=None, percentile=68.0)

        Determine the fit upper and lower constraint given a percentile.

        Parameters
        ----------
        bestfit : BootstrapFit
            Best-fit parameter used to evaluate the model.
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
        # resample the covariance matrix and evaluate the model for each sample
        param_samples = np.array([
            self(z, *p) for p in np.random.multivariate_normal(
                bestfit.paramBest(), bestfit.paramCovar(),
                size=bestfit.n_samples)])
        # compute the range of the percentiles of the model distribution
        p = (100.0 - percentile) / 2.0
        n_z_min, n_z_max = np.percentile(param_samples, [p, 100.0 - p], axis=0)
        return z, n_z_min, n_z_max

    def plot(self, bestfit, z, ax=None, **kwargs):
        """
        plot(self, ax=None, **kwargs)

        Create an error bar plot the data sample.

        Parameters
        ----------
        bestfit : BootstrapFit
            Best-fit parameter used to evaluate the model.
        z : array_like
            Redshifts at with the model is sampled.
        ax : matplotlib.axes
            Specifies the axis to plot on.
        **kwargs : keyword arguments
            Arugments parsed on to matplotlib.pyplot.plot and fill_between

        Returns
        -------
        bestfit : BootstrapFit
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
    PowerLawBias()

    Simple bias model of the form (1 + z)^a.
    """

    n_param = 1

    def guess(self):
        """
        Parameter guess for the amplitudes in the format required by
        scipy.optmize.curve_fit.
        """
        return np.zeros(self.n_param)

    def bounds(self):
        """
        Parameter bounds for the amplitudes in the format required by
        scipy.optmize.curve_fit.
        """
        return (np.full(self.n_param, -5.0), np.full(self.n_param, 5.0))

    def __call__(self, z, *params):
        """
        PowerLawBias(z, *params)

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

    def autoSampling(self):
        """
        Automatic sampling of the model based on the spread of the components.
        """
        z = np.linspace(0.0, self.mus[-1] + 3.0 * self.sigmas[-1], 200)
        return z

    def modelBest(self, bestfit, z=None):
        """
        modelBest(bestfit, z=None)

        Evaluate the fit model with the best fit parameters.

        Parameters
        ----------
        bestfit : BootstrapFit
            Best-fit parameter used to evaluate the model.
        z : array_like
            Redshifts at with the model is sampled. If None, this is calculated
            from the redshift range spanned by the model components.

        Returns
        -------
        z : array_like
            Redshifts at with the model is sampled.
        n_z : array_like
            Model value.
        """
        if z is None:
            z = self.autoSampling()
        z, n_z = super().modelBest(bestfit, z)
        return z, n_z

    def modelError(self, bestfit, z=None, percentile=68.0):
        """
        modelError(z=None, percentile=68.0)

        Determine the fit upper and lower constraint given a percentile.

        Parameters
        ----------
        bestfit : BootstrapFit
            Best-fit parameter used to evaluate the model.
        z : array_like
            Redshifts at with the model is sampled. If None, this is calculated
            from the redshift range spanned by the model components.
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
        if z is None:
            z = self.autoSampling()
        z, n_z_min, n_z_max = super().modelError(bestfit, z, percentile)
        return z, n_z_min, n_z_max

    def plot(self, bestfit, z=None, ax=None, **kwargs):
        """
        plot(self, ax=None, **kwargs)

        Create an error bar plot the data sample.

        Parameters
        ----------
        bestfit : BootstrapFit
            Best-fit parameter used to evaluate the model.
        z : array_like
            Redshifts at with the model is sampled. If None, this is calculated
            from the redshift range spanned by the model components.
        ax : matplotlib.axes
            Specifies the axis to plot on.
        **kwargs : keyword arguments
            Arugments parsed on to matplotlib.pyplot.plot and fill_between

        Returns
        -------
        bestfit : BootstrapFit
            Parameter best-fit container.
        """
        if z is None:
            z = self.autoSampling()
        super().plot(bestfit, z, ax, **kwargs)


class GaussianComb(CombModel):
    """
    GaussianComb(n_param, z0, dz, smoothing=1.0)

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
        """
        Parameter guess for the amplitudes in the format required by
        scipy.optmize.curve_fit.
        """
        return np.ones(self.n_param)

    def bounds(self):
        """
        Parameter bounds for the amplitudes in the format required by
        scipy.optmize.curve_fit.
        """
        return (np.full(self.n_param, 0.0), np.full(self.n_param, np.inf))

    def __call__(self, z, *params):
        """
        GaussianComb(z, *params)

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
    LogGaussianComb(n_param, z0, dz, smoothing=1.0)

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
        """
        Parameter guess for the amplitudes in the format required by
        scipy.optmize.curve_fit.
        """
        return np.zeros(self.n_param)

    def __call__(self, z, *params):
        """
        LogGaussianComb(z, *params)

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


class MultiBinModel(BaseModel):

    def __init__(self, bins, weights):
        assert(all(isinstance(m, BaseModel) for m in bins))
        assert(len(weights) == len(bins))
        self._models = [m for m in bins]
        self.weights = np.asarray(weights) / sum(weights)
        self.n_models = len(bins)
        self.n_param_model = [m.n_param for m in bins]
        self.n_param = sum(self.n_param_model)

    def guess(self):
        """
        Parameter guess for the amplitudes in the format required by
        scipy.optmize.curve_fit.
        """
        guess = np.concatenate([m.guess() for m in self._models])
        return guess

    def bounds(self):
        """
        Parameter bounds for the amplitudes in the format required by
        scipy.optmize.curve_fit.
        """
        lower_bound = np.concatenate([m.bounds()[0] for m in self._models])
        upper_bound = np.concatenate([m.bounds()[1] for m in self._models])
        return (lower_bound, upper_bound)

    def __call__(self, z, *params):
        """
        Factory method for the model implementation.
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

    def autoSampling(self):
        """
        Factory method to automatically sample the model.
        """
        samplings = [m.autoSampling() for m in self._models]
        z = np.concatenate(samplings)
        master_bin = np.linspace(0.0, z.max(), len(samplings[0]))
        return np.append(z, master_bin)

    def modelBest(self, bestfit, z=None):
        """
        modelBest(bestfit, z=None)

        Evaluate the fit model with the best fit parameters.

        Parameters
        ----------
        bestfit : BootstrapFit
            Best-fit parameter used to evaluate the model.
        z : array_like
            Redshifts at with the model is sampled. If None, this is calculated
            from the redshift range spanned by the model components.

        Returns
        -------
        z : array_like
            Redshifts at with the model is sampled.
        n_z : array_like
            Model value.
        """
        if z is None:
            z = self.autoSampling()
        z, n_z = super().modelBest(bestfit, z)
        return z, n_z

    def modelError(self, bestfit, z=None, percentile=68.0):
        """
        modelError(z=None, percentile=68.0)

        Determine the fit upper and lower constraint given a percentile.

        Parameters
        ----------
        bestfit : BootstrapFit
            Best-fit parameter used to evaluate the model.
        z : array_like
            Redshifts at with the model is sampled. If None, this is calculated
            from the redshift range spanned by the model components.
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
        if z is None:
            z = self.autoSampling()
        z, n_z_min, n_z_max = super().modelError(bestfit, z, percentile)
        return z, n_z_min, n_z_max

    def plot(self, bestfit, z=None, fig=None, **kwargs):
        """
        plot(self, ax=None, **kwargs)

        Create an error bar plot the data sample.

        Parameters
        ----------
        bestfit : BootstrapFit
            Best-fit parameter used to evaluate the model.
        z : array_like
            Redshifts at with the model is sampled. If None, this is calculated
            from the redshift range spanned by the model components.
        ax : matplotlib.axes
            Specifies the axis to plot on.
        **kwargs : keyword arguments
            Arugments parsed on to matplotlib.pyplot.plot and fill_between

        Returns
        -------
        bestfit : BootstrapFit
            Parameter best-fit container.
        """
        if fig is None:
            # try to arrange the subplots in a grid
            n_x = int(np.ceil(self.n_data / np.sqrt(self.n_data)))
            n_y = int(np.ceil(self.n_data / n_x))
            fig, axes = plt.subplots(
                n_y, n_x, figsize=(4 * n_x, 4 * n_y), sharex=True, sharey=True)
        else:
            axes = np.asarray(fig.axes)
        if z is None:
            z = self.autoSampling()
        plot_kwargs = {}
        plot_kwargs.update(kwargs)
        # compute the plot values
        bin_z = np.split(z, self.n_models + 1)
        _, n_z = self.modelBest(bestfit, z)
        bin_n_z = np.split(n_z, self.n_models + 1)
        _, n_z_min, n_z_max = self.modelError(bestfit, z)
        bin_n_z_min = np.split(n_z_min, self.n_models + 1)
        bin_n_z_max = np.split(n_z_max, self.n_models + 1)
        for i, ax in enumerate(axes.flatten()):
            line = ax.plot(bin_z[i], bin_n_z[i], **plot_kwargs)[0]
            # add a shaded area that indicates the 68% model confidence
            try:
                plot_kwargs.pop("color")
            except KeyError:
                pass
            ax.fill_between(
                bin_z[i], bin_n_z_min[i], bin_n_z_max[i], alpha=0.3,
                color=line.get_color(), **plot_kwargs)
