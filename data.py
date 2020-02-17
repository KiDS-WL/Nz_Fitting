from collections import OrderedDict
from copy import copy
import os

import numpy as np
import numpy.ma as ma
import pandas as pd
from corner import corner
from matplotlib import cm as colormaps
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d

from .utils import Figure, format_variable

DEFAULT_EXT_DATA = ".dat"
DEFAULT_EXT_BOOT = ".boot"
DEFAULT_EXT_COV = ".cov"


class BaseData:

    def _parse_method(self, method):
        if method is None:
            method = self.samplingMethod()
        elif method not in ("samples", "covmat", "stderr"):
            raise ValueError("invalid method name '{:}'".format(method))
        return method

    def iterSamples(self, limit=1000, method=None):
        method = self._parse_method(method)
        if self.hasSamples() and method == "samples":
            limit = self.getSampleNo()
        for idx in range(limit):
            yield self.getSample(idx)

    def plotCorrMat(self, all=True, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        corrmat = self.getCorrMat(all=all, concat=True)
        # create space for a new axis hosting the color map
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        # represent NaNs by light grey
        cmap = colormaps.bwr
        cmap.set_bad(color="0.7")
        # plot and add color map to new axis
        im = ax.imshow(corrmat, vmin=-1.0, vmax=1.0, cmap=cmap)
        plt.gcf().colorbar(im, cax=cax, orientation="vertical")


class RedshiftData(BaseData):

    def __init__(self, z, n, dn=None, weight=1.0):
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
        self.setWeight(weight)

    def _update_masks(self, extra_mask=None):
        # check which values are unmasked in all data elements
        mask = self._z.mask | self._n.mask | self._dn.mask
        if self.hasCovMat():
            mask |= np.diag(self._covmat.mask)
        if mask.all():
            raise ValueError("data has only invalid values")
        # update each data vector mask
        for attr in ("_z", "_n", "_dn"):
            getattr(self, attr).mask = mask
        # apply to covariance matrix
        if self.hasCovMat():
            self._covmat.mask[mask, :] = True
            self._covmat.mask[:, mask] = True

    @staticmethod
    def read(
            basepath, ext_dat=DEFAULT_EXT_DATA,
            ext_boot=DEFAULT_EXT_BOOT, ext_cov=DEFAULT_EXT_COV):
        file_dat = basepath + ext_dat
        if not os.path.exists(file_dat):
            raise OSError("input data file '{:}' not found".format(file_dat))
        # load data and create a RedshiftData instance
        data = np.loadtxt(file_dat)
        if len(data.shape) != 2:
            raise ValueError("expected 2-dim data file")
        data = RedshiftData(*data.T[:3])
        # try loading samples or a covariance matrix
        if os.path.exists(basepath + ext_boot):
            samples = np.loadtxt(basepath + ext_boot)
            data.setSamples(samples)
        # if we have samples we do not need the covariance anymore
        elif os.path.exists(basepath + ext_cov):
            covmat = np.loadtxt(basepath + ext_cov)
            data.setCovMat(covmat)
        return data

    def write(
            self, basepath, head_dat=None,
            head_boot=None, head_cov=None, ext_dat=DEFAULT_EXT_DATA,
            ext_boot=DEFAULT_EXT_BOOT, ext_cov=DEFAULT_EXT_COV):
        # write the data
        data = np.stack([self.z(True), self.n(True), self.dn(True)]).T
        if head_dat is None:
            head_dat = (
                "col 1 = redshift\n" +
                "col 2 = fraction at redshift\n" +
                "col 3 = error of fraction")
        np.savetxt(basepath + ext_dat, data, header=head_dat, fmt="% 9.6f")
        # write the samples
        if self.hasSamples():
            if head_boot is None:
                head_boot = "samples of fraction at redshift"
            np.savetxt(
                basepath + ext_boot, self.getSamples(all=True),
                header=head_boot, fmt="% 9.6f")
        # write the covariance matrix
        if self.hasCovMat():
            if head_cov is None:
                head_cov = "covariance matrix of fraction at redshift"
            np.savetxt(
                basepath + ext_cov, self.getCovMat(all=True),
                header=head_cov, fmt="% 12.5e")

    def mask(self, **kwargs):
        return self._z.mask

    def len(self, all=False, **kwargs):
        return len(self._z) if all else ma.count(self._z)
 
    def z(self, all=False, **kwargs):
        return self._z.filled(np.nan) if all else self._z.compressed()

    def n(self, all=False, **kwargs):
        return self._n.filled(np.nan) if all else self._n.compressed()

    def dn(self, all=False, **kwargs):
        return self._dn.filled(np.nan) if all else self._dn.compressed()

    def setWeight(self, weight):
        self._weight = np.float64(weight)

    def getWeight(self):
        return self._weight

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

    def setEdges(self, edges):
        n_bins = self.len(all=True) + 1
        if edges.shape != (n_bins,):
            raise ValueError(
                "expected bin edges of shape ({:d},), ".format(n_bins) + 
                "but got shape {:s}".format(str(edges.shape)))
        if np.any(np.diff(edges) <= 0.0):
            raise ValueError("bins must increase monotonically")
        self._edges = np.array(edges)

    def hasEdges(self, require=False):
        has_edges = hasattr(self, "_edges")
        if require and not has_edges:
            raise AttributeError("redshift bin edges not set")
        return has_edges

    def getEdges(self):
        self.hasEdges(require=True)
        return self._edges

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

    def getSamples(self, all=False, **kwargs):
        self.hasSamples(require=True)
        if all:
            samples = self._samples.data
        else:
            samples = []
            for sample in self._samples:
                samples.append(sample.compressed())
        return samples

    def getSampleNo(self):
        self.hasSamples(require=True)
        return len(self._samples)

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
        self._covmat = covmat
        self._update_masks()
        cov_diag = np.diag(self._covmat)
        if np.all(self.dn() == 1.0):
            self.setErrors(np.sqrt(cov_diag))
        else:
            variance = self.dn() ** 2
            if not np.isclose(cov_diag.compressed(), variance).all():
                raise ValueError(
                    "variance and covariance matrix diagonal do not match")

    def hasCovMat(self, require=False):
        has_covmat = hasattr(self, "_covmat")
        if require and not has_covmat:
            raise AttributeError("covariance matrix not set")
        return has_covmat

    def getCovMat(self, all=False, **kwargs):
        self.hasCovMat(require=True)
        if all:
            covmat = self._covmat.filled(np.nan)
        else:
            n_good = self.len()
            covmat = self._covmat.compressed().reshape((n_good, n_good))
        return covmat

    def getCorrMat(self, all=False, **kwargs):
        covmat = self.getCovMat(all=False)
        # normalize the matrix elements
        norm = np.sqrt(np.diag(covmat))
        corrmat = covmat / np.outer(norm, norm)
        if all:
            # get the bad columns and merge them with the correlation matrix
            corrmat_full = self._covmat.copy()
            corrmat_full[corrmat_full != ma.masked] = corrmat.flatten()
            return corrmat_full.filled(np.nan)
        else:
            return corrmat

    def getCovMatInv(self, all=False, **kwargs):
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

    def getSample(self, idx=None, method=None):
        method = self._parse_method(method)
        if method == "samples" and idx is not None:
            n_samples = self.getSampleNo()
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
        if self.hasCovMat():
            new.setCovMat(self.getCovMat(all=True))
        if self.hasEdges():
            new.setEdges(self.getEdges())
        return new

    def norm(self):
        norm = np.trapz(self.n(), x=self.z())
        return norm if norm >= 0.0 else np.nan

    def pdf(self, all=False, **kwargs):
        return self.n(all) / self.norm()

    def cdf(self, all=False, **kwargs):
        cdf = cumtrapz(self.n(), x=self.z(), initial=0.0)
        cdf /= cdf[-1] if cdf[-1] > 0.0 else np.nan  # normalisation
        if all:
            cdf_full = self._n.copy()
            cdf_full[cdf_full != ma.masked] = cdf
            return cdf_full.filled(np.nan)
        else:
            return cdf

    def mean(self, error=False):
        z = self.z()
        mean = np.trapz(z * self.pdf(), x=z)
        if error:
            # compute the error from realisations
            means = np.fromiter(
                (sample.mean(error=False) for sample in self.iterSamples()),
                dtype=np.dtype(mean))
            return mean, np.nanstd(means)
        else:
            return mean

    def median(self, error=False):
        # median: z where cdf(z) == 0.5
        median = np.interp(0.5, self.cdf(), self.z())  # returns redshift
        if error:
            # compute the error from realisations
            medians = np.fromiter(
                (sample.median(error=False) for sample in self.iterSamples()),
                dtype=np.dtype(median))
            return median, np.nanstd(medians)
        else:
            return median

    def plotPoints(self, ax=None, z_offset=0.0, mark_edges=True, **kwargs):
        if ax is None:
            fig = Figure(1)
            ax = fig.axes[0]
        else:
            fig = plt.gcf()
        plot_kwargs = {"color": "k", "marker": ".", "ls": "none"}
        plot_kwargs.update(kwargs)
        if mark_edges and self.hasEdges():
            edges = self.getEdges()
            ax.plot(
                edges, np.zeros_like(edges),
                color=plot_kwargs["color"], marker="|", ls="none")
        ax.errorbar(
            self.z() + z_offset, self.n(), yerr=self.dn(), **plot_kwargs)
        return fig

    def plotHist(self, ax=None, **kwargs):
        if ax is None:
            fig = Figure(1)
            ax = fig.axes[0]
        else:
            fig = plt.gcf()
        pdf = self.pdf(all=True)
        y = np.append(pdf[0], pdf)
        fill_kwargs = {}
        if "color" not in kwargs:
            kwargs["color"] = "0.6"
        if "label" in kwargs:
            fill_kwargs["label"] = kwargs.pop("label")
        fill = ax.fill_between(
            self.getEdges(), 0.0, np.nan_to_num(y), 
            step="pre", alpha=0.3, **fill_kwargs)
        lines = ax.step(self.getEdges(), y, **kwargs)
        fill.set_color(lines[0].get_color())
        return fig


class RedshiftDataBinned(BaseData):

    def __init__(self, bins, master):
        self._data = [*bins, master]
        for data in self._data:
            if not isinstance(data, RedshiftData):
                raise TypeError(
                    "'bins' and 'master' must be of type 'RedshiftData'")
        # check same number of samples
        if self.hasSamples():
            n_samples = [data.getSampleNo() for data in self._data]
            if not all(n_samples[0] == n for n in n_samples[1:]):
                raise ValueError(
                    "Number of data samples does not match in input")

    def __len__(self):
        return len(self._data)

    def _collect(self, attr, *args, callback=None, apply=False):
        items = [getattr(data, attr)(*args) for data in self._data]
        return callback(items) if apply else items

    def mask(self, concat=False):
        return self._collect("mask", callback=np.concatenate, apply=concat)

    def len(self, all=False, concat=False):
        return self._collect("len", all, callback=sum, apply=concat)

    def z(self, all=False, concat=False):
        return self._collect("z", all, callback=np.concatenate, apply=concat)

    def n(self, all=False, concat=False):
        return self._collect("n", all, callback=np.concatenate, apply=concat)

    def dn(self, all=False, concat=False):
        return self._collect("dn", all, callback=np.concatenate, apply=concat)

    def getWeight(self):
        weights = [data.getWeight() for data in self._data]
        master = weights.pop()
        if not np.isclose(sum(weights), master):
            raise ValueError("bin weights do not add up to master weight")
        return [w / master for w in [*weights, master]]

    def getEdges(self, concat=False):
        return self._collect("getEdges", callback=np.concatenate, apply=concat)

    def iterData(self):
        for data in self._data:
            yield data

    def iterBins(self):
        for data in self._data[:-1]:
            yield data

    def getMaster(self):
        return self._data[-1]

    def assertEqualZ(self):
        zs = self.z()
        ref_z = zs.pop()
        for z in zs:
            assert(np.all(z == ref_z))

    def hasSamples(self, require=False):
        return all(data.hasSamples(require) for data in self._data)

    def getSampleNo(self):
        self.hasSamples(require=True)
        return self._data[0].getSampleNo()

    def getSamples(self, all=False, concat=False):
        self.hasSamples(require=True)
        bin_samples = [data.getSamples(all) for data in self._data]
        if concat:
            if all:
                samples = np.concatenate(bin_samples, axis=1)
            else:
                samples = []
                for sample_idx in range(self.getSampleNo()):
                    samples.append(
                        np.concatenate([
                            bin_samples[bin_idx][sample_idx]
                            for bin_idx in range(len(self))]))
            return samples
        else:
            return bin_samples

    def hasCovMat(self, require=False):
        return all(data.hasCovMat(require) for data in self._data)

    @staticmethod
    def _block_matrix(diagonal_blocks, fill_value=0.0):
        shapes = [
            [b.shape[0] for b in diagonal_blocks],
            [b.shape[1] for b in diagonal_blocks]]
        blocks = []
        for i0, shape0 in enumerate(shapes[0]):
            row = []
            for i1, shape1 in enumerate(shapes[1]):
                if i0 == i1:
                    row.append(diagonal_blocks[i0])
                else:
                    row.append(np.full((shape0, shape1), fill_value))
            blocks.append(row)
        matrix = np.block(blocks)
        # fill missing NaNs everywhere
        mask = np.isnan(np.diag(matrix))
        matrix[mask, :] = np.nan
        matrix[:, mask] = np.nan
        return matrix

    def getCovMat(self, all=False, concat=False):
        return self._collect(
            "getCovMat", all, callback=self._block_matrix, apply=concat)

    def getCorrMat(self, all=False, concat=False):
        return self._collect(
            "getCorrMat", all, callback=self._block_matrix, apply=concat)

    def getCovMatInv(self, all=False, concat=False):
        return self._collect(
            "getCovMatInv", all, callback=self._block_matrix, apply=concat)

    def samplingMethod(self):
        methods = [data.samplingMethod() for data in self._data]
        if all(method == "samples" for method in methods):
            return "samples"
        elif all(method != "stderr" for method in methods):
            return "covmat"
        else:
            return "stderr"

    def getSample(self, idx=None, method=None):
        method = self._parse_method(method)
        samples = [data.getSample(idx, method=method) for data in self._data]
        new = self.__class__(samples[:-1], samples[-1])
        return new

    def norm(self):
        return [data.norm() for data in self._data]

    def pdf(self, all=False, concat=False):
        return self._collect("pdf", all, callback=np.concatenate, apply=concat)

    def cdf(self, all=False, concat=False):
        return self._collect("cdf", all, callback=np.concatenate, apply=concat)

    def mean(self, error=False):
        means = [data.mean(error) for data in self._data]
        if error:
            return [m[0] for m in means], [m[1] for m in means]
        else:
            return means

    def median(self, error=False):
        medians = [data.median(error) for data in self._data]
        if error:
            return [m[0] for m in medians], [m[1] for m in medians]
        else:
            return medians

    def _get_fig(self, fig):
        if fig is None:
            fig = Figure(len(self))
            axes = np.asarray(fig.axes)
        else:
            axes = np.asarray(fig.axes)
        return fig, axes

    def plotPoints(self, fig=None, z_offset=0.0, mark_edges=True, **kwargs):
        fig, axes = self._get_fig(fig)
        # plot data sets the axes and delete the remaining ones from the grid
        for i, ax in enumerate(axes.flatten()):
            self._data[i].plotPoints(
                ax=ax, z_offset=z_offset, mark_edges=mark_edges, **kwargs)
        return fig

    def plotHist(self, fig=None, **kwargs):
        fig, axes = self._get_fig(fig)
        # plot data sets the axes and delete the remaining ones from the grid
        for i, ax in enumerate(axes.flatten()):
            self._data[i].plotHist(ax=ax, **kwargs)
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
