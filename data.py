import numpy as np
from corner import corner
from matplotlib import pyplot as plt


class DataTuple(object):
    """
    DataTuple(z, n, dn)

    Container for redshift distribution data with uncertainties.

    Parameters
    ----------
    z : array_like
        Sampling points of the redshift distribution.
    n : array_like
        Redshift distribution.
    dn : array_like
        Redshift distribution standard error.
    """

    cov = None

    def __init__(self, z, n, dn):
        self.z = np.asarray(z)
        assert(len(self) == len(n))
        assert(len(self) == len(dn))
        self.n = np.asarray(n)
        self.dn = np.asarray(dn)

    def __len__(self):
        return len(self.z)

    def setCovariance(self, cov):
        """
        setCovariance(cov)

        Add an optional data covariance matrix.

        Parameters
        ----------
        cov : array_like
            covariance matrix.
        """
        cov = np.asarray(cov)
        assert(cov.shape == (len(self), len(self), ))
        self.cov = cov

    def resample(self):
        """
        resample()

        Resample the data based on the standard error or the covariance matrix.

        Parameters
        ----------
        new : DataTuple
            Copy of the DataTuple instance with resampled redshift
            distribution.
        """
        if self.cov is None:
            n = np.random.normal(self.n, self.dn)
            new = DataTuple(self.z, n, self.dn)
        else:
            n = np.random.multivariate_normal(self.n, self.cov)
            new = DataTuple(self.z, n, self.dn)
            new.setCovariance(self.cov)
        return new

    def plot(self, ax=None, **kwargs):
        """
        plot(self, ax=None, **kwargs)

        Create an error bar plot the data sample.

        Parameters
        ----------
        ax : matplotlib.axes
            Specifies the axis to plot on.
        **kwargs : keyword arguments
            Arugments parsed on to matplotlib.pyplot.errorbar

        Returns
        -------
        bestfit : BootstrapFit
            Parameter best-fit container.
        """
        if ax is None:
            ax = plt.gca()
        plot_kwargs = {"color": "k", "marker": ".", "ls": "none"}
        plot_kwargs.update(kwargs)
        ax.errorbar(self.z, self.n, yerr=self.dn, **plot_kwargs)


class MultiBinData(DataTuple):

    def __init__(self, bins, master):
        assert(all(isinstance(b, DataTuple) for b in bins))
        assert(isinstance(master, DataTuple))
        self._data = [*bins, master]
        self.n_data = len(self._data)
        # all input samples must have the same redshift sampling
        assert(all(np.all(d.z == master.z) for d in self._data))
        # assemble all data points into a vector
        self.z = np.concatenate([d.z for d in self._data])
        self.n = np.concatenate([d.n for d in self._data])
        self.dn = np.concatenate([d.dn for d in self._data])

    def plot(self, fig=None, **kwargs):
        if fig is None:
            # try to arrange the subplots in a grid
            n_x = int(np.ceil(self.n_data / np.sqrt(self.n_data)))
            n_y = int(np.ceil(self.n_data / n_x))
            fig, axes = plt.subplots(
                n_y, n_x, figsize=(4 * n_x, 4 * n_y), sharex=True, sharey=True)
        else:
            axes = np.asarray(fig.axes)
        # plot data sets the axes and delete the remaining ones from the grid
        for i, ax in enumerate(axes.flatten()):
            try:
                self._data[i].plot(ax=ax, **kwargs)
            except IndexError:
                fig.delaxes(axes.flatten()[i])
        fig.tight_layout()
        return fig


class BootstrapFit(object):
    """
    BootstrapFit(bestfit, fitsamples)

    Container for best-fit parameters and and samples for correlatin
    estimation.

    Parameters
    ----------
    bestfit : array_like
        List of best-fit model parameters.
    fitsamples : array_like
        Samples of the fit parameters.
    """

    def __init__(self, bestfit, fitsamples):
        self._best = np.asarray(bestfit)
        self._samples = np.asarray(fitsamples)
        self.n_samples, self.n_param = self._samples.shape
        assert(self.n_param == len(bestfit))

    def paramBest(self):
        """
        Get the best fit parameters.
        """
        return self._best.copy()

    def paramError(self):
        """
        Get the best fit parameter errors.
        """
        variance = np.diag(self.paramCovar())
        return np.sqrt(variance)

    def paramCovar(self):
        """
        Get the best fit parameter covariance matrix.
        """
        cov = np.cov(self._samples, rowvar=False)
        return np.atleast_2d(cov)

    def paramCorr(self):
        """
        Get the best fit parameter correlation matrix.
        """
        errors = self.paramError()
        covar = self.paramCovar()
        corr = np.matmul(
            np.matmul(np.diag(1.0 / errors), covar), np.diag(1.0 / errors))
        return corr

    def plotSamples(self):
        """
        Plot the distribution of the fit parameter samples in a triangle plot
        with corner.corner.
        """
        return corner(self._samples)

    def plotCorr(self):
        """
        Plot the correlation matrix.
        """
        im = plt.matshow(self.paramCorr(), vmin=-1, vmax=1, cmap="bwr")
        plt.colorbar(im)
