import os
import warnings
from collections import OrderedDict
from copy import copy

import numpy as np
import numpy.ma as ma
from matplotlib import cm as colormaps
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d

from .utils import Figure


DEFAULT_EXT_HIST = ".hist"
DEFAULT_EXT_DATA = ".dat"
DEFAULT_EXT_BOOT = ".boot"
DEFAULT_EXT_COV = ".cov"


class Base:

    def _optimizerGetZ(self, attr):
        assert(attr in ("z", "edges"))
        return getattr(self, attr)(all=False, concat=False)


class BaseData(Base):

    _covmat_check = False
    _covmat_default = "global"

    def _parseMethod(self, method):
        if method is None:
            method = self.samplingMethod()
        elif method not in ("samples", "covmat", "stderr"):
            raise ValueError("invalid method name '{:}'".format(method))
        return method

    def iterSamples(self, limit=1000, method=None):
        method = self._parseMethod(method)
        if self.hasSamples() and method == "samples":
            limit = self.getSampleNo()
        for idx in range(limit):
            yield self.getSample(idx)

    @staticmethod
    def _warnCovMat(message, handle="warn"):
        assert(handle in ("print", "warn", "error", "ignore"))
        if handle == "error":
            raise ValueError(message)
        elif handle == "warn":
            warnings.warn(message)
        elif handle == "print":
            print("WARNING: " + message)

    def setCovMat(self, covmat, check=True):
        covmat = ma.masked_invalid(covmat)
        # check whether data and covariance dimensions match
        n_data = self.len(all=True, concat=True)
        if covmat.shape != (n_data, n_data):
            raise ValueError(
                "expected covariance matrix of shape " +
                "({n:d}, {n:d}), ".format(n=n_data) + 
                "but got shape {:s}".format(covmat.shape))
        if check:
            # check if it is positive definite
            mask = ~self.mask(concat=True) & ~np.diag(covmat.mask)
            n_data = np.count_nonzero(mask)
            new_mask_2D = np.ix_(mask, mask)
            if self._covmat_check:
                np.linalg.cholesky(covmat[new_mask_2D])
            # do some basic checks with the diagonal
            cov_diag = np.diag(covmat)[mask]
            variance = self.dn(all=True, concat=True)[mask] ** 2
            """ temporarily disabled
            if not np.isclose(cov_diag, variance).all():
                raise ValueError(
                    "variance and covariance matrix diagonal do not match")
            """
        self._covmat = covmat
        self._updateMasks()

    def _parseCovMatType(self, ctype):
        if ctype is not None:
            if ctype not in self._covmat_types:
                raise ValueError(
                    "'ctype' must be either of ({:}), not '{:}'".format(
                        ", ".join(self._covmat_types), ctype))
        else:
            ctype = self.getCovMatType()
        return ctype

    def setCovMatType(self, ctype):
        if ctype is None:
            ctype = "none"  # parse an invalid value to raise an exception
        self._covmat_default = self._parseCovMatType(ctype)

    def getCovMatType(self):
        return copy(self._covmat_default)

    def getCovMat(self, all=False, ctype=None, inv=False):
        self.hasCovMat(require=True)
        covmat_type = self._parseCovMatType(ctype)
        # we want all entries for inversion
        if all and (not inv):
            if covmat_type == "global":
                covmat = self._covmat.filled(np.nan)
            else:
                variance = self.dn(all=True, concat=True) ** 2
                covmat = ma.diag(variance)
                covmat.mask = self._covmat.mask.copy()
                covmat = covmat.filled(np.nan)
        else:
            if covmat_type == "global":
                n_good = self.len(concat=True)
                covmat = self._covmat.compressed().reshape((n_good, n_good))
            else:
                variance = self.dn(all=False, concat=True) ** 2
                covmat = np.diag(variance)
        if inv:
            # compute the inverse of the covariance matrix with good columns
            invmat_good = np.linalg.inv(covmat)
            if all:
                # get the bad columns and merge them with the inverse matrix
                invmat = self._covmat.copy()
                invmat[invmat != ma.masked] = invmat_good.flatten()
                invmat = invmat.filled(np.nan)
            else:
                invmat = invmat_good
            return invmat
        else:
            return covmat

    def getCorrMat(self, all=False, ctype=None):
        covmat = self.getCovMat(all=False, ctype=ctype)
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

    def plotCovMat(self, all=True, ctype=None, inv=False, ax=None):
        if ax is None:
            ax = plt.gca()
        covmat = self.getCovMat(all=all, ctype=ctype, inv=inv)
        # create space for a new axis hosting the color map
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.0)
        # represent NaNs by light grey
        nan_color = "0.7"
        cmap = colormaps.bwr
        cmap.set_bad(color=nan_color)
        ax.set_facecolor(nan_color)
        # plot and add color map to new axis
        vabs = np.nanmax(np.abs(covmat))
        im = ax.imshow(covmat, vmin=-vabs, vmax=vabs, cmap=cmap)
        plt.gcf().colorbar(im, cax=cax, orientation="vertical")

    def plotCorrMat(self, all=True, ctype=None, ax=None):
        if ax is None:
            ax = plt.gca()
        corrmat = self.getCorrMat(all=all, ctype=ctype)
        # create space for a new axis hosting the color map
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.0)
        # represent NaNs by light grey
        nan_color = "0.7"
        cmap = colormaps.bwr
        cmap.set_bad(color=nan_color)
        ax.set_facecolor(nan_color)
        # plot and add color map to new axis
        im = ax.imshow(corrmat, vmin=-1.0, vmax=1.0, cmap=cmap)
        plt.gcf().colorbar(im, cax=cax, orientation="vertical")


class BaseBinned(Base):

    def __len__(self):
        return len(self._data)

    def _collect(self, attr, *args, callback=None, apply=False):
        items = [getattr(data, attr)(*args) for data in self._data]
        return callback(items) if apply else items

    def assertEqual(self, data):
        ref = data.pop()
        for d in data:
            assert(np.all(d == ref))

    def iterData(self):
        for data in self._data:
            yield data

    def iterBins(self):
        for data in self._data[:-1]:
            yield data

    def getBins(self):
        return self._data[:-1]

    def getMaster(self):
        return self._data[-1]

    def getData(self):
        return self._data

    def norm(self):
        return [data.norm() for data in self._data]

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


class RedshiftHistogram(Base):

    def __init__(self, edges, counts, centers=None):
        self._edges = np.array(edges)
        if np.any(np.diff(self._edges) <= 0.0):
            raise ValueError(
                "redshift bin edges must increase monotonically")
        self._n = np.array(counts)
        if np.any(counts < 0.0):
            raise ValueError("'counts' must note be negative")
        if centers is None:
            self._z = (self._edges[1:] + self._edges[:-1]) / 2.0
        else:
            self._z = np.array(centers)
        edge_shape = (self._edges.shape[0] - 1, *self._edges.shape[1:])
        if not (edge_shape == self._n.shape == self._z.shape):
            raise ValueError(
                "length of edges ({:d} - 1), ".format(len(edges)) +
                "counts ({:d}) and ".format(len(counts)) +
                "centers ({:d}) do not match".format(len(self._z)))
        # check that the centers can belong to the edges
        if centers is not None:
            idx_insert = np.searchsorted(self._edges, self._z)
            if np.any(np.diff(idx_insert) != 1):
                raise ValueError(
                    "every bin center must fall into one subsequent bin")
            if np.any(idx_insert < 1) or np.any(idx_insert > edge_shape[0]):
                raise ValueError(
                    "bin centers extend below or above the binning")
        self._makePdf()
        self._makeCdf()

    @staticmethod
    def read(basepath, ext=DEFAULT_EXT_HIST):
        if not os.path.exists(basepath + ext):
            raise OSError(
                "input data file '{:}' not found".format(basepath + ext))
        # load data and create a RedshiftHistogram instance
        data = np.loadtxt(basepath + ext)
        if len(data.shape) != 2:
            raise ValueError("expected 2-dim data file")
        edges = data[:, 0]
        counts = data[:-1, 1]  # the last one is a fill value
        if data.shape[1] > 2:
            centers = data[:-1, 2]  # the last one is a fill value
        else:
            centers = None
        hist = RedshiftHistogram(edges, counts, centers)
        return hist

    def write(self, basepath, head=None, ext=DEFAULT_EXT_HIST):
        # write the data
        data = np.zeros((len(self._edges), 3))
        data[:, 0] = self._edges
        data[:-1, 1] = self._n  # the last one is a fill value
        data[:-1, 2] = self._z  # the last one is a fill value
        if head is None:
            head = (
                "col 1 = bin edges\n" +
                "col 2 = counts\n" +
                "col 3 = bin centers")
        np.savetxt(basepath + ext, data, header=head, fmt="% 12.5e")

    def len(self, **kwargs):
        return len(self._n)

    def edges(self, **kwargs):
        return self._edges.copy()

    def counts(self, **kwargs):
        return self._n.copy()

    def centers(self, **kwargs):
        return self._z.copy()

    def norm(self):
        return np.sum(self._n * np.diff(self._edges))

    def _makePdf(self):
        self._pdf = self._n / self.norm()

    def pdf(self, z, **kwargs):
        # look up the histogram values at the given redshift
        idx = np.digitize(z, self._edges[1:-1])
        pdf = self._pdf[idx]
        # extrapolate zeros
        pdf[z < self._edges[0]] = 0.0
        pdf[z >= self._edges[-1]] = 0.0
        return pdf

    def _makeCdf(self):
        cdf_edges = np.append(
            0.0, np.cumsum(self._n * np.diff(self._edges)))
        cdf_edges /= cdf_edges[-1]  # normalize
        self._cdf_spline = interp1d(
            self._edges, cdf_edges,
            fill_value=(0.0, 1.0), bounds_error=False)

    def cdf(self, z, **kwargs):
        return self._cdf_spline(z)

    def mean(self, **kwargs):
        # interpret pdf as step function
        pdf = self.pdf(self._edges[:-1])
        pdf = np.append(pdf, pdf[-1])
        # compute integral over pdf * z
        pdf_times_z = pdf * self._edges
        pdf_times_z_integral = (pdf_times_z[1:] + pdf_times_z[:-1]) / 2.0
        pdf_times_z_integral *= np.diff(self._edges)
        mean = np.sum(pdf_times_z_integral)
        return mean

    def median(self, **kwargs):
        # median: z where cdf(z) == 0.5
        cdf = self.cdf(self._edges)
        median = np.interp(0.5, cdf, self._edges)  # returns redshift
        return median

    def plot(self, ax=None, **kwargs):
        if ax is None:
            fig = Figure(1)
            ax = fig.axes[0]
        else:
            fig = plt.gcf()
        y = np.append(self._pdf[0], self._pdf)
        fill_kwargs = {}
        if "color" not in kwargs:
            kwargs["color"] = "0.6"
        if "label" in kwargs:
            fill_kwargs["label"] = kwargs.pop("label")
        fill = ax.fill_between(
            self._edges, 0.0, np.nan_to_num(y), 
            step="pre", alpha=0.3, **fill_kwargs)
        lines = ax.step(self._edges, y, **kwargs)
        fill.set_color(lines[0].get_color())
        return fig


class RedshiftData(BaseData):

    _covmat_types = ("diagonal", "global")

    def __init__(self, z, n, dn):
        self._z = ma.masked_invalid(z)
        if np.any(np.diff(self._z.compressed()) <= 0.0):
            raise ValueError(
                "redshift sampling points must increase monotonically")
        self._n = ma.masked_invalid(n)
        self._dn = ma.masked_invalid(dn)
        if not (self._z.shape == self._n.shape == self._dn.shape):
            raise ValueError(
                "length of z ({:d}), ".format(len(z)) +
                "n ({:d}) and ".format(len(n)) +
                "dn ({:d}) do not match".format(len(dn)))
        # synchronize the individual masks
        self._updateMasks()

    def _updateMasks(self, extra_mask=None):
        # check which values are unmasked in all data elements
        mask = self._z.mask | self._n.mask | self._dn.mask
        # identify data points with unreasonable SNR
        """ for later use
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mask |= np.abs(self._n / self._dn) > 1e9
        """
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
                basepath + ext_cov, self.getCovMat(all=True, ctype="global"),
                header=head_cov, fmt="% 12.5e")

    def mask(self, **kwargs):
        return self._z.mask.copy()

    def len(self, all=False, **kwargs):
        return len(self._z) if all else ma.count(self._z)
 
    def z(self, all=False, **kwargs):
        return self._z.filled(np.nan) if all else self._z.compressed()

    def n(self, all=False, **kwargs):
        return self._n.filled(np.nan) if all else self._n.compressed()

    def dn(self, all=False, **kwargs):
        return self._dn.filled(np.nan) if all else self._dn.compressed()

    def setErrors(self, errors):
        errors = ma.masked_invalid(errors)
        n_data = self.len(all=True)
        if errors.shape != (n_data,):
            raise ValueError(
                "expected errors of shape ({:d},), ".format(n_data) + 
                "but got shape {:s}".format(str(errors.shape)))
        self._dn = errors
        # synchronize the individual masks
        self._updateMasks()

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

    def edges(self, **kwargs):
        self.hasEdges(require=True)
        return self._edges.copy()

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
        try:
            self.setCovMat(covmat)
        except np.linalg.LinAlgError:
            self._warnCovMat(
                "automatically estimated covariance matrix is not positive " +
                "definite")

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

    def hasCovMat(self, require=False):
        has_covmat = hasattr(self, "_covmat")
        if require and not has_covmat:
            raise AttributeError("covariance matrix not set")
        return has_covmat

    def samplingMethod(self):
        if self.hasSamples():
            return "samples"
        elif self.hasCovMat():
            return "covmat"
        else:
            return "stderr"

    def getSample(self, idx=None, method=None):
        method = self._parseMethod(method)
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
        # set the same covariance state
        new.setCovMatType(self.getCovMatType())
        if self.hasCovMat():
            new.setCovMat(self.getCovMat(all=True), check=False)
        if self.hasEdges():
            new.setEdges(self.edges())
        return new

    def norm(self):
        norm = np.trapz(self.n(), x=self.z())
        return norm if norm >= 0.0 else np.nan

    def pdf(self, all=False, **kwargs):
        return self.n(all=all) / self.norm()

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

    def normalize(self):
        """
        normalize the stored values
        """
        norm = self.norm()
        self._n /= norm
        self._dn /= norm
        if self.hasSamples():
            self.setSamples(self._samples / norm)
        elif self.hasCovMat():
            self.setCovMat(self._covmat / norm**2)

    def plot(self, ax=None, lines=False, z_offset=0.0, **kwargs):
        if ax is None:
            fig = Figure(1)
            ax = fig.axes[0]
        else:
            fig = plt.gcf()
        if lines:
            line = ax.plot(self.z(), self.n(), **kwargs)[0]
            if np.any(self.dn() != 0.0):
                ax.fill_between(
                    self.z(), self.n() - self.dn(), self.n() + self.dn(),
                    alpha=0.3, color=line.get_color())
        else:
            plot_kwargs = {"color": "k", "marker": ".", "ls": "none"}
            plot_kwargs.update(kwargs)
            ax.errorbar(
                self.z() + z_offset, self.n(), yerr=self.dn(), **plot_kwargs)
        return fig


class RedshiftHistogramBinned(BaseBinned):

    def __init__(self, bins, master):
        self._data = [*bins, master]
        for data in self._data:
            if not isinstance(data, RedshiftHistogram):
                raise TypeError(
                    "'bins' and 'master' must be of type 'RedshiftHistogram'")

    def len(self, concat=False, **kwargs):
        return self._collect("len", callback=sum, apply=concat)

    def edges(self, concat=False, **kwargs):
        return self._collect("edges", callback=np.concatenate, apply=concat)

    def counts(self, concat=False, **kwargs):
        return self._collect("counts", callback=np.concatenate, apply=concat)

    def centers(self, concat=False, **kwargs):
        return self._collect("centers", callback=np.concatenate, apply=concat)

    def pdf(self, z, concat=False, **kwargs):
        pdfs = []
        for bin_z, data in zip(z, self._data):
            pdfs.append(data.pdf(bin_z))
        if concat:
            return np.concatenate(pdfs)
        else:
            return pdfs

    def cdf(self, z, concat=False, **kwargs):
        cdfs = []
        for bin_z, data in zip(z, self._data):
            cdfs.append(data.cdf(bin_z))
        if concat:
            return np.concatenate(cdfs)
        else:
            return cdfs

    def mean(self, **kwargs):
        return [data.mean() for data in self._data]

    def median(self, **kwargs):
        return [data.median() for data in self._data]

    def plot(self, fig=None, **kwargs):
        fig, axes = self._getFig(fig)
        for i, ax in enumerate(axes.flatten()):
            self._data[i].plot(ax=ax, **kwargs)
        return fig


class RedshiftDataBinned(BaseData, BaseBinned):

    _covmat_types = ("diagonal", "blockdiag", "block", "global")

    def __init__(self, bins, master):
        self._data = [*bins, master]
        for data in self._data:
            if not isinstance(data, RedshiftData):
                raise TypeError(
                    "'bins' and 'master' must be of type 'RedshiftData'")
        if self.hasSamples():
            # check same number of samples
            n_samples = [data.getSampleNo() for data in self._data]
            if not all(n_samples[0] == n for n in n_samples[1:]):
                raise ValueError(
                    "Number of data samples does not match in input")
            # compute the global covariance
            samples = ma.masked_invalid(self.getSamples(all=True, concat=True))
            covmat = ma.cov(samples, ddof=0, rowvar=False)
            try:
                self.setCovMat(covmat)
            except np.linalg.LinAlgError:
                self._warnCovMat(
                    "global covariance matrix is not positive definite, "
                    "falling back to block diagonal matrix of bin covariances")

    def _updateMasks(self):
        mask = self.mask(concat=True)
        # apply to covariance matrix
        if self.hasCovMat():
            self._covmat.mask[mask, :] = True
            self._covmat.mask[:, mask] = True

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

    def edges(self, concat=False, **kwargs):
        return self._collect("edges", callback=np.concatenate, apply=concat)

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
        has_covmat = hasattr(self, "_covmat")
        if not has_covmat:
            has_covmat = all(data.hasCovMat(require) for data in self._data)
        if require and not has_covmat:
            raise AttributeError(
                "neither individual nor global covariance matrix not set")
        return has_covmat

    @staticmethod
    def _blockMatrix(diagonal_blocks, fill_value=0.0):
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

    def getCovMat(self, all=False, ctype=None, inv=False):
        covmat_type = self._parseCovMatType(ctype)
        if covmat_type in ("global", "diagonal"):
            return super().getCovMat(all=all, ctype=covmat_type, inv=inv)
        elif covmat_type == "blockdiag":
            covmat = super().getCovMat(all=False, ctype="global")
            main_diag = np.diag(covmat)
            # set everything except the (block-)diagonals to zero
            block_cov_good = np.diag(main_diag)
            off_diag_idx = np.cumsum(self.len(all=False))
            for k in off_diag_idx:
                off_diag = np.diag(covmat, k)
                # use symmetry
                block_cov_good += np.diag(off_diag, k)
                block_cov_good += np.diag(off_diag, -k)
            # reinsert NaNs
            if all and (not inv):
                block_cov = self._covmat.copy()
                block_cov[block_cov != ma.masked] = block_cov_good.flatten()
                block_cov = block_cov.filled(np.nan)
            else:
                block_cov = block_cov_good
            if inv:
                # compute the inverse of the covariance matrix with good columns
                block_inv_good = np.linalg.inv(block_cov_good)
                if all:
                    # get the bad columns and merge them with the inverse matrix
                    block_inv = self._covmat.copy()
                    block_inv[block_inv != ma.masked] = block_inv_good.flatten()
                    block_inv = block_inv.filled(np.nan)
                else:
                    block_inv = block_inv_good
                return block_inv
            else:
                return block_cov
        else:
            return self._blockMatrix(
                self._collect("getCovMat", all, None, inv))

    def getCorrMat(self, all=False, ctype=None):
        covmat_type = self._parseCovMatType(ctype)
        if covmat_type in ("global", "diagonal"):
            return super().getCorrMat(all=all, ctype=covmat_type)
        elif covmat_type == "blockdiag":
            covmat = self.getCovMat(all=all, ctype=ctype)
            norm = np.sqrt(np.diag(covmat))
            return covmat / np.outer(norm, norm)
        else:
            return self._blockMatrix(self._collect("getCorrMat", all, None))

    def writeCovMat(self, basepath, head=None, ext=DEFAULT_EXT_COV):
        if head is None:
            head = "cross-bin covariance matrix of fraction at redshift"
        np.savetxt(
            basepath + ext, self.getCovMat(all=True, ctype="global"),
            header=head, fmt="% 12.5e")

    def samplingMethod(self):
        methods = [data.samplingMethod() for data in self._data]
        if all(method == "samples" for method in methods):
            return "samples"
        elif all(method != "stderr" for method in methods):
            return "covmat"
        else:
            return "stderr"

    def getSample(self, idx=None, method=None):
        method = self._parseMethod(method)
        samples = [data.getSample(idx, method=method) for data in self._data]
        new = self.__class__(samples[:-1], samples[-1])
        # set the same covariance state
        new.setCovMatType(self.getCovMatType())
        if hasattr(self, "_covmat"):  # copy global covariance
            new.setCovMat(
                self.getCovMat(all=True, ctype="global"), check=False)
        return new

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

    def plot(self, fig=None, lines=False, z_offset=0.0, **kwargs):
        fig, axes = self._getFig(fig)
        for i, ax in enumerate(axes.flatten()):
            self._data[i].plot(
                ax=ax, lines=lines, z_offset=z_offset, **kwargs)
        return fig


def load_KiDS_bins(scaledir_path, normalize=False):
    bin_data = []
    try:
        for zbin in (
                "0.101z0.301", "0.301z0.501", "0.501z0.701",
                "0.701z0.901", "0.901z1.201", "0.101z1.201"):
            for prefix in ("/crosscorr_", "/shiftfit_"):
                try:
                    data = RedshiftData.read(scaledir_path + prefix + zbin)
                    if normalize:
                        data.normalize()
                    bin_data.append(data)
                except OSError:
                    pass
        bins = RedshiftDataBinned(bin_data[:-1], bin_data[-1])
        if not bins.hasSamples():
            global_covmat_path = os.path.join(
                scaledir_path, "crosscorr_global.cov")
            bins.setCovMat(np.loadtxt(global_covmat_path))
    except IndexError:
        for zbin in (
                "0.101z0.301", "0.301z0.501", "0.501z0.701",
                "0.701z0.901", "0.901z1.201", "0.101z1.201"):
            for f in os.listdir(scaledir_path):
                if f.endswith("%s.hist" % zbin):
                    bin_data.append(
                        RedshiftHistogram.read(os.path.join(
                            scaledir_path, os.path.splitext(f)[0])))
        bins = RedshiftHistogramBinned(bin_data[:-1], bin_data[-1])
    return bins
