from collections import OrderedDict
from copy import copy

import numpy as np
import pandas as pd
from corner import corner
from matplotlib import pyplot as plt
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d

from .utils import format_variable, Figure


class RedshiftHistogram(object):

    def __init__(self, bin_centers, counts):
        self.z = np.asarray(bin_centers)
        assert(len(self) == len(counts))
        # store the normalized counts
        self.pdf = np.asarray(counts) / np.trapz(counts, x=self.z)
        self.cdf = cumtrapz(self.pdf, x=self.z, initial=0.0)

    def __len__(self):
        return len(self.z)

    def plot(self, ax=None, **kwargs):
        """
        Create an error bar plot the data histogram.

        Parameters
        ----------
        ax : matplotlib.axes
            Specifies the axis to plot on.
        **kwargs : keyword arguments
            Arugments parsed on to matplotlib.pyplot.step
        """
        if ax is None:
            fig = Figure(1)
            ax = plt.gca()
        else:
            fig = plt.gcf()
        y = np.append(self.pdf[0], self.pdf)
        fill_kwargs = {}
        if "color" not in kwargs:
            kwargs["color"] = "0.6"
        if "label" in kwargs:
            fill_kwargs["label"] = kwargs.pop("label")
        fill = ax.fill_between(
            self.z, 0.0, y,
            step="pre", alpha=0.3, **fill_kwargs)
        lines = ax.step(self.z, y, **kwargs)
        fill.set_color(lines[0].get_color())
        return fig


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
    reals = None

    def __init__(self, z, n, dn=None):
        self.z = np.array(z, copy=True)
        assert(len(self) == len(n))
        assert(len(self) == len(dn))
        self.n = np.array(n, copy=True)
        self.dn = np.array(dn, copy=True)
        self.dn[np.isnan(self.dn)] = np.inf

    def __len__(self):
        return len(self.z)

    def setCovariance(self, cov, check=True):
        """
        Add an optional data covariance matrix.

        Parameters
        ----------
        cov : array_like
            Covariance matrix of shape (N data x N data).
        """
        cov = np.asarray(cov)
        if check:
            if not cov.shape == (len(self), len(self), ):
                raise ValueError(
                    ("data vector has length %d, but " % len(self)) +
                    "covariance matrix has shape %s" % str(cov.shape))
            var = self.dn**2
            if not np.isclose(np.diag(cov), var).all():
                string = "variance and diagonal of covariance matrix "
                string += "do not match:\n"
                string += "variance:   %s\n" % str(var)
                string += "cov. diag.: %s\n" % str(np.diag(cov))
                raise ValueError(string)
        self.cov = cov

    def getCovariance(self):
        if self.cov is None:
            raise AttributeError("covariance matrix not set")
        else:
            return self.cov

    def getInvserseCovariance(self):
        if self.cov_inv is None:
            self.cov_inv = np.linalg.inv(self.getCovariance())
        return self.cov_inv

    def setRealisations(self, n_array):
        """
        Add an optional data realisations.

        Parameters
        ----------
        n_array : array_like
            Data vector realisations of shape (N realisations x N data).
        """
        reals = np.asarray(n_array)
        if not n_array.shape[1] == len(self):
            raise ValueError(
                ("data vector has length %d, but realisations " % len(self)) +
                "have length %s" % str(n_array.shape[1]))
        self.reals = reals

    def getRealisations(self):
        if self.reals is None:
            raise AttributeError("realisations not set")
        else:
            return self.reals


    def getNoRealisations(self):
        if self.reals is None:
            return 0
        else:
            return self.reals.shape[0]

    def mean(self, n=None):
        n = self.n if n is None else n
        mask = np.isfinite(n)
        z, n = self.z[mask], n[mask]
        norm = np.trapz(n, x=z)
        return np.trapz(z * n / norm, x=z)

    def meanError(self, n_samples=1000):
        means = []
        if self.reals is None:
            for i in range(n_samples):
                means.append(self.resample().mean())
        else:
            for i in range(len(self.reals)):
                means.append(self.resample(i).mean())
        return np.std(means, axis=0)

    def median(self, n=None):
        n = self.n if n is None else n
        mask = np.isfinite(n)
        z, n = self.z[mask], n[mask]
        cdf = cumtrapz(n, x=z, initial=0.0)
        cdf /= cdf[-1]  # normalize
        # median: z where cdf(z) == 0.5
        cdf_inverse = interp1d(cdf, z)  # returns redshift
        return np.float64(cdf_inverse(0.5))  # median

    def medianError(self, n_samples=1000):
        medians = []
        if self.reals is None:
            for i in range(n_samples):
                medians.append(self.resample().median())
        else:
            for i in range(len(self.reals)):
                medians.append(self.resample(i).median())
        return np.std(medians, axis=0)

    def plotCorr(self):
        """
        Plot the correlation matrix.
        """
        corr = np.matmul(
            np.matmul(np.diag(1.0 / self.dn), self.cov),
            np.diag(1.0 / self.dn))
        im = plt.matshow(corr, vmin=-1, vmax=1, cmap="bwr")
        plt.colorbar(im)

    def resample(self, reals_idx=None):
        """
        If data vector realisations exist, one of these can be selected,
        otherwise resample the data based on the standard error or the
        covariance matrix.

        Parameters
        ----------
        reals_idx : int
            Index of data vector realisation. If None (default), generate a
            random realisations based on the covariance matrix or standard
            errors.

        Returns
        -------
        new : RedshiftData
            Copy of the RedshiftData instance with resampled redshift
            distribution.
        """
        # invalid situation
        if self.reals is None and reals_idx is not None:
            raise ValueError("no realisations found to draw from")
        # get specific realisation
        elif self.reals is not None and reals_idx is not None:
            new = self.__class__(self.z, self.reals[reals_idx], self.dn)
        # draw random realisations
        elif reals_idx is None and self.cov is not None:
            n = np.random.multivariate_normal(self.n, self.cov)
            new = self.__class__(self.z, n, self.dn)
        else:
            n = np.random.normal(self.n, self.dn)
            new = self.__class__(self.z, n, self.dn)
        # copy over covariance matrix, but not the realisations
        if self.cov is not None:
            new.setCovariance(self.cov, check=False)
        return new

    def plot(self, ax=None, z_offset=0.0, **kwargs):
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
            fig = Figure(1)
            ax = plt.gca()
        else:
            fig = plt.gcf()
        plot_kwargs = {"color": "k", "marker": ".", "ls": "none"}
        plot_kwargs.update(kwargs)
        ax.errorbar(self.z + z_offset, self.n, yerr=self.dn, **plot_kwargs)
        return fig


class RedshiftDataBinned(RedshiftData):
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
        self.data = [*bins, master]
        self.n_data = len(self.data)
        # assemble all data points into a vector
        self.z = np.concatenate([d.z for d in self.data])
        self.n = np.concatenate([d.n for d in self.data])
        self.dn = np.concatenate([d.dn for d in self.data])
        # check if there are any realisations
        if any(d.reals is None for d in self.data):
            self.reals = None
        else:
            self.reals = np.concatenate([d.reals for d in self.data], axis=1)

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

    def mean(self):
        return np.array([d.mean() for d in self.data])

    def median(self):
        return np.array([d.median() for d in self.data])

    def resample(self, reals_idx=None):
        """
        If data vector realisations exist, one of these can be selected,
        otherwise resample the data based on the standard error or the
        covariance matrix.

        Parameters
        ----------
        reals_idx : int
            Index of data vector realisation. If None (default), generate a
            random realisations based on the covariance matrix or standard
            errors.

        Returns
        -------
        new : BinnedRedshiftData
            Copy of the BinnedRedshiftData instance with resampled redshift
            distribution.
        """
        # invalid situation
        if self.reals is None and reals_idx is not None:
            raise KeyError("no realisations found to draw from")
        # get specific realisation
        elif self.reals is not None and reals_idx is not None:
            n = self.reals[reals_idx]
        # draw random realisations
        elif reals_idx is None and self.cov is None:
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

    def plot(self, fig=None, z_offset=0.0, **kwargs):
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
            fig = Figure(self.n_data)
            axes = np.asarray(fig.axes)
        else:
            axes = np.asarray(fig.axes)
        # plot data sets the axes and delete the remaining ones from the grid
        for i, ax in enumerate(axes.flatten()):
            bins[i].plot(ax=ax, z_offset=z_offset, **kwargs)
        return fig


class FitParameters(object):
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
