from copy import copy

import numpy as np
from corner import corner
from matplotlib import pyplot as plt
from scipy.integrate import cumtrapz


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
            ax = plt.gca()
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
            Covariance matrix of shape (N data x N data).
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

    def getNoRealisations(self):
        if self.reals is None:
            return 0
        else:
            return self.reals.shape[0]

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
            raise KeyError("no realisations found to draw from")
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
        data = [*bins, master]
        self.n_data = len(data)
        # assemble all data points into a vector
        self.z = np.concatenate([d.z for d in data])
        self.n = np.concatenate([d.n for d in data])
        self.dn = np.concatenate([d.dn for d in data])
        # check if there are any realisations
        if any(d.reals is None for d in data):
            self.reals = None
        else:
            self.reals = np.concatenate([d.reals for d in data], axis=1)

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
        elif reals_idx is None and self.cov is not None:
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

    def __init__(self, bestfit, fitsamples, model):
        self._best = np.asarray(bestfit)
        self._samples = np.asarray(fitsamples)
        self.n_samples = len(self._samples)
        self._n_param = model.getParamNo()
        self._param_names = model.getParamNames(label=False)
        self._param_labels = model.getParamNames(label=True)
        assert(self._n_param == len(bestfit))

    def __len__(self):
        return self.n_samples

    def __str__(self):
        max_width = max(len(n) for n in self.getParamNames())
        string = ""
        iterator = zip(
            self.getParamNames(), self.paramBest(), self.paramError())
        for name, value, error in iterator:
            string += "{:>{w}} = {:}\n".format(
                name, self._format_value_error(value, error, precision=3),
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
        return copy(self._n_param)

    def getParamNames(self, label=False):
        """
        Names of the free model parameters.
        """
        if label:
            return copy(self._param_labels)
        else:
            return copy(self._param_names)

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

    @staticmethod
    def _format_value_error(value, error, precision, TEX=False):
        exponent_value = "{:.{sign}e}".format(value, sign=precision)
        exponent_error = "{:.{sign}e}".format(error, sign=precision)
        exponent = max(
            int(exponent_value.split("e")[1]),
            int(exponent_error.split("e")[1]))
        # decide which formatter to use
        if -3 < exponent < 3:
            expression = "${: {dig}.{sign}f} {:} {:{dig}.{sign}f}$".format(
                value, "\pm" if TEX else "±", error,
                dig=precision + 2, sign=precision)
            if not TEX:
                expression = expression.strip("$")
        else:
            norm = 10 ** exponent
            if TEX:
                expression = "$({:.{sign}f} \pm {:.{sign}f}) ".format(
                    value / norm, error / norm, sign=precision)
                expression += "\\times 10^{{{:}}}$".format(exponent)
            else:
                expression = "({: .{sign}f} ± {:.{sign}f}) x 1e{e:d}".format(
                    value / norm, error / norm, e=exponent, sign=precision)
        return expression

    def paramAsTEX(self, param_name, precision=3):
        """
        TODO
        """
        precision = max(0, precision)
        if param_name not in self._param_names:
            raise KeyError("parameter name '%s' does not exist")
        # get the data values
        idx = self._param_names.index(param_name)
        label = self._param_labels[idx]
        value = self.paramBest()[idx]
        error = self.paramError()[idx]
        # format to TEX, decide automatically which formatter to use
        expression = self._format_value_error(value, error, precision)
        TEXstring = "${:} = {:}$".format(
            label.strip("$"), expression.strip("$"))
        return TEXstring

    def plotSamples(self):
        """
        Plot the distribution of the fit parameter samples in a triangle plot
        with corner.corner.
        """
        fig = corner(
            self._samples, labels=self.getParamNames(label=True),
            show_titles=True)
        fig.tight_layout()
        return fig

    def plotCorr(self):
        """
        Plot the correlation matrix.
        """
        im = plt.matshow(self.paramCorr(), vmin=-1, vmax=1, cmap="bwr")
        plt.colorbar(im)
