from collections import OrderedDict
from copy import copy

import numpy as np
import numpy.ma as ma
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

    def __init__(self, z, n, dn=None, weight=1.0):
        self.weight = float(weight)
        self._z = ma.masked_invalid(z)
        self._n = ma.masked_invalid(n)
        if dn is None:  # create a dummmy array
            self._dn = ma.array(np.ones_like(self._z))
        else:
            self._dn = ma.masked_invalid(dn)
        if not (self._z.shape == self._n.shape == self._dn.shape):
            raise ValueError(
                "length of z ({:d}), ".format(len(z)) +
                "n ({:d}) and ".format(len(n)) +
                "dn ({:d}}) do not match".format(len(dn)))
        # synchronize the individual masks
        self._update_masks()

    def _update_masks(self):
        mask = self._z.mask | self._n.mask | self._dn.mask
        if mask.all():
            raise ValueError("data has only invalid values")
        for attr in ("_z", "_n", "_dn"):  # update each mask
            getattr(self, attr).mask = mask

    @staticmethod
    def read(path):
        NotImplemented

    def write(self, path):
        NotImplemented

    def mask(self):
        return self._z.mask

    def len(self, all=False, **kwargs):
        return len(self._z) if all else ma.count(self._z)
 
    def z(self, all=False, **kwargs):
        return self._z.data if all else self._z.compressed()

    def n(self, all=False, **kwargs):
        return self._n.data if all else self._n.compressed()

    def dn(self, all=False, **kwargs):
        return self._dn.data if all else self._dn.compressed()

    def setErrors(self, errors):
        errors = ma.masked_invalid(errors)
        n_data = self.len(all=True)
        if errors.shape != (n_data,):
            raise ValueError(
                "expected errors of shape ({:d},), ".format(n_data) + 
                "but got shape {:s}".format(str(errors.shape)))
        self._dn = errors
        # synchronize the individual masks
        self._update_masks()

    def setSamples(self, samples):
        samples = ma.masked_invalid(samples)
        n_data = self.len(all=True)
        # check whether data and sample dimensions match
        try:
            assert(samples.shape[1] == n_data)
        except (IndexError, AssertionError):
            raise ValueError(
                "expected samples of shape (N, {:d}), ".format(n_data) + 
                "but got shape {:s}".format(str(samples.shape)))
        # check that no realisations are completely masked
        if np.any(samples.mask.sum(axis=1) == n_data):
            raise ValueError("a sample contains no valid values")
        self._samples = samples
        # replace the existing error bars
        self.setErrors(samples.std(axis=0))
        # set the covariance matrix
        covmat = ma.cov(self._samples, ddof=0, rowvar=False)
        self.setCovMat(covmat)

    def hasSamples(self, require=False):
        has_samples = hasattr(self, "_samples")
        if require and not has_samples:
            raise AttributeError("data samples not set")
        return has_samples

    def getSamples(self, all=False):
        self.hasSamples(require=True)
        if all:
            samples = self._samples.data
        else:
            samples = []
            for sample in self._samples:
                samples.append(sample.compressed())
        return samples

    def setCovMat(self, covmat):
        covmat = ma.masked_invalid(covmat)
        # check whether data and covariance dimensions match
        n_data = self.len(all=True)
        if covmat.shape != (n_data, n_data):
            raise ValueError(
                "expected covariance matrix of shape " +
                "({n:d}, {n:d}), ".format(n=n_data) + 
                "but got shape {:s}".format(covmat.shape))
        # do some basic checks with the diagonal
        diag = np.diag(covmat)
        if not np.all(diag.mask == self.mask()):
            raise ValueError(
                "the mask of data and covariance matrix diagonal do not match")
        if not np.isclose(diag.compressed(), self.dn()**2).all():
            raise ValueError(
                "the variance and the covariance matrix diagonal do not match")
        self._covmat = covmat

    def hasCovMat(self, require=False):
        has_covmat = hasattr(self, "_covmat")
        if require and not has_covmat:
            raise AttributeError("covariance matrix not set")
        return has_covmat

    def getCovMat(self, all=False):
        self.hasCovMat(require=True)
        if all:
            covmat = self._covmat.filled(np.nan)
        else:
            n_good = self.len()
            covmat = self._covmat.compressed().reshape((n_good, n_good))
        return covmat

    def getCovMatInv(self, all=False):
        if not hasattr(self, "_covmat_inv"):
            # compute the inverse of the covariance matrix with good columns
            covmat_good = self.getCovMat(all=False)
            invmat_good = np.linalg.inv(covmat_good)
            # get the bad columns and merge them with the inverse matrix
            invmat = self._covmat.copy()
            invmat[invmat != ma.masked] = invmat_good.flatten()
            self._invmat = invmat
        if all:
            invmat = self._invmat.filled(np.nan)
        else:
            n_good = self.len()
            invmat = self._invmat.compressed().reshape((n_good, n_good))
        return invmat

    def samplingMethod(self):
        if self.hasSamples():
            return "samples"
        elif self.hasCovMat():
            return "covmat"
        else:
            return "stderr"

    def getSample(self, idx=None):
        method = self.samplingMethod()
        if method == "samples" and idx is not None:
            n_samples = self._samples.shape[0]
            if idx >= n_samples:
                raise IndexError(
                    "requested sample index {:d} is out of range {:d}".format(
                        idx, n_samples))
            sample = self._samples[idx]
        else:
            sample = self._n.copy()
            if method == "covmat":
                sample[sample != ma.masked] = np.random.multivariate_normal(
                    self.n(all=False), self.getCovMat(all=False))
            else:
                sample[sample != ma.masked] = np.random.normal(
                    self.n(all=False), self.dn(all=False))
        new = self.__class__(self.z(all=True), sample.data, self.dn(all=True))
        return new

    def iterSamples(self, limit=1000):
        if self.hasSamples():
            limit = self._samples.shape[0]
        for idx in range(limit):
            yield self.getSample(idx)

    def norm(self):
        return np.trapz(self.n(), x=self.z())

    def mean(self):
        z = self.z()
        return np.trapz(z * self.n() / self.norm(), x=z)

    def mean(self, error=False):
        z = self.z()
        mean = np.trapz(z * self.n() / self.norm(), x=z)
        if error:
            # compute the error
            means = [
                sample.mean(error=False) for sample in self.iterSamples()]
            return mean, np.std(means)
        else:
            return mean

    def median(self, error=False):
        z = self.z()
        cdf = cumtrapz(self.n(), x=z, initial=0.0)
        cdf /= cdf[-1]  # normalize
        # median: z where cdf(z) == 0.5
        median = np.interp(0.5, cdf, z)  # returns redshift
        if error:
            # compute the error
            medians = [
                sample.median(error=False) for sample in self.iterSamples()]
            return median, np.std(medians)
        else:
            return median

    def plot(self, ax=None, z_offset=0.0, mark_bad=False, **kwargs):
        if ax is None:
            fig = Figure(1)
            ax = fig.axes[0]
        else:
            fig = plt.gcf()
        plot_kwargs = {"color": "k", "marker": ".", "ls": "none"}
        plot_kwargs.update(kwargs)
        if mark_bad:
            z_bad = self._z[self.mask()].data
            ax.plot(
                z_bad, np.zeros_like(z_bad),
                color=plot_kwargs["color"], marker="|", ls="none")
        ax.errorbar(
            self.z() + z_offset, self.n(), yerr=self.dn(), **plot_kwargs)
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
