from copy import copy

import numpy as np
from corner import corner
from matplotlib import pyplot as plt


class RedshiftData(object):
    """
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
        Add an optional data covariance matrix.

        Parameters
        ----------
        cov : array_like
            covariance matrix.
        """
        cov = np.asarray(cov)
        if not cov.shape == (len(self), len(self), ):
            raise ValueError(
                ("data vector has length %d, but covariance " % len(self)) +
                "matrix has shape %s" % str(cov.shape))
        if not np.isclose(np.diag(cov), self.dn**2).all():
            raise ValueError(
                "variance and diagonal of covariance matrix do not match")
        self.cov = cov

    def plotCorr(self):
        """
        Plot the correlation matrix.
        """
        corr = np.matmul(
            np.matmul(np.diag(1.0 / self.dn), self.cov),
            np.diag(1.0 / self.dn))
        im = plt.matshow(corr, vmin=-1, vmax=1, cmap="bwr")
        plt.colorbar(im)

    def resample(self):
        """
        Resample the data based on the standard error or the covariance matrix.

        Returns
        -------
        new : RedshiftData
            Copy of the RedshiftData instance with resampled redshift
            distribution.
        """
        if self.cov is None:
            n = np.random.normal(self.n, self.dn)
            new = self.__class__(self.z, n, self.dn)
        else:
            n = np.random.multivariate_normal(self.n, self.cov)
            new = self.__class__(self.z, n, self.dn)
            new.setCovariance(self.cov)
        return new

    def plot(self, ax=None, **kwargs):
        """
        Create an error bar plot the data sample.

        Parameters
        ----------
        ax : matplotlib.axes
            Specifies the axis to plot on.
        **kwargs : keyword arguments
            Arugments parsed on to matplotlib.pyplot.errorbar
        """
        if ax is None:
            ax = plt.gca()
        plot_kwargs = {"color": "k", "marker": ".", "ls": "none"}
        plot_kwargs.update(kwargs)
        ax.errorbar(self.z, self.n, yerr=self.dn, **plot_kwargs)


class BinnedRedshiftData(RedshiftData):
    """
    Container a joint tomographic bin fitting. The weighted sum of the bins
    is fitted against the full sample (master) and this container bundles all
    the redshift distributions for such a fit.

    Parameters
    ----------
    bins : array_like of RedshiftData
        Set of data from tomographic bins as RedshiftData.
    master : RedshiftData
        Data of the full sample redshift distribution.
    """

    def __init__(self, bins, master):
        assert(all(isinstance(b, RedshiftData) for b in bins))
        assert(isinstance(master, RedshiftData))
        # all input samples must have the same redshift sampling
        assert(all(np.all(d.z == master.z) for d in bins))
        data = [*bins, master]
        self.n_data = len(data)
        # assemble all data points into a vector
        self.z = np.concatenate([d.z for d in data])
        self.n = np.concatenate([d.n for d in data])
        self.dn = np.concatenate([d.dn for d in data])

    def split(self):
        """
        Split the data vector back into a list of bin data samples

        Returns
        -------
        bins : list of RedshiftData
            Data split into tomographic bins with the full sample in the last
            position
        """
        binned_z = np.split(self.z, self.n_data)
        binned_n = np.split(self.n, self.n_data)
        binned_dn = np.split(self.dn, self.n_data)
        bins = [
            RedshiftData(z, n, dn)
            for z, n, dn in zip(binned_z, binned_n, binned_dn)]
        return bins

    def resample(self):
        """
        Resample the data based on the standard error or the covariance matrix.

        Returns
        -------
        new : BinnedRedshiftData
            Copy of the BinnedRedshiftData instance with resampled redshift
            distribution.
        """
        if self.cov is None:
            n = np.random.normal(self.n, self.dn)
        else:
            n = np.random.multivariate_normal(self.n, self.cov)
        # split the data back into the individual data sets
        binned_z = np.split(self.z, self.n_data)
        binned_n = np.split(n, self.n_data)
        binned_dn = np.split(self.dn, self.n_data)
        bins = [
            RedshiftData(z, n, dn)
            for z, n, dn in zip(binned_z, binned_n, binned_dn)]
        new = self.__class__(bins[:-1], bins[-1])
        # add the missing class members
        new.n_data = self.n_data
        if self.cov is not None:
            new.setCovariance(self.cov)
        return new

    def plot(self, fig=None, **kwargs):
        """
        Create an error bar plot the data sample. Tomographic bins are arranged
        in a grid of separate plots followed by the (full) master sample.

        Parameters
        ----------
        fig : matplotlib.figure
            Plot on an existig figure which must have at least n_data axes.
        **kwargs : keyword arguments
            Arugments parsed on to matplotlib.pyplot.errorbar

        Returns
        -------
        fig : matplotlib.figure
            The figure containting the plots.
        """
        bins = self.split()
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
                bins[i].plot(ax=ax, **kwargs)
            except IndexError:
                fig.delaxes(axes.flatten()[i])
        fig.tight_layout()
        return fig


class FitParameters(object):
    """
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

    def paramSamples(self):
        """
        Directly get the fit parameter samples.
        """
        return self._samples.copy()

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
