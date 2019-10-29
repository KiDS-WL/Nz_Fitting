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
        plt.errorbar(self.z, self.n, yerr=self.dn, **plot_kwargs)


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
        return np.cov(self._samples, rowvar=False)

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
