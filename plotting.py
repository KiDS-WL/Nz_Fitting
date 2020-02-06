import numpy as np
from matplotlib import pyplot as plt


class Figure:

    def __init__(self, n_plots, dpi=150):
        # try to arrange the subplots in a grid
        n_x = max(1, int(np.ceil(n_plots / np.sqrt(n_plots))))
        n_y = max(1, int(np.ceil(n_plots / n_x)))
        self.shape = (n_x, n_y)
        self.fig, axes = plt.subplots(
            n_y, n_x, figsize=(
                0.5 + 3.5 * n_x, 0.5 + 3.0 * n_y),
            sharex=True, sharey=True, dpi=dpi)
        for i, ax in enumerate(self.axes):
            ax.grid(alpha=0.2)
            if i < n_plots:
                ax.tick_params(
                    "both", direction="in",
                    bottom=True, top=True, left=True, right=True)
            else:
                self.delaxes(ax)
        self.set_facecolor("white")

    def no_gaps(self):
        self.tight_layout(h_pad=0.0, w_pad=0.0)
        self.subplots_adjust(top=0.92, hspace=0.0, wspace=0.0)

    def no_margins(self, x=None, y=None):
        for ax in self.axes:
            ax.margins(x=x, y=y)

    def set_xlabel(self, text, fontsize=14):
        for i, ax in enumerate(self.axes):
            if i // self.shape[0] == self.shape[1] - 1:
                ax.set_xlabel(text, fontsize=fontsize)

    def set_ylabel(self, text, fontsize=14):
        for i, ax in enumerate(self.axes):
            if i % self.shape[0] == 0:
                ax.set_ylabel(text, fontsize=fontsize)

    def set_xlim(self, left=None, right=None):
        for ax in self.axes:
            ax.set_xlim(left, right)

    def set_ylim(self, bottom=None, top=None):
        for ax in self.axes:
            ax.set_ylim(bottom, top)

    def annotate(self, text_list, loc, xycoords="axes fraction", size=None):
        if type(text_list) is str:
            text_list = [text_list] * len(self.axes)
        for ax, text in zip(self.axes, text_list):
            ax.annotate(
                text, loc, xycoords=xycoords,
                va="center", ha="center", size=None)

    def vspans(self, limit_tuples, **kwargs):
        if "color" not in kwargs:
            kwargs["color"] = "k"
        if "alpha" not in kwargs:
            kwargs["alpha"] = 0.1
        for ax, limits in zip(self.axes, limit_tuples):
            ax.axvspan(*limits, **kwargs)

    def __getattr__(self, attr):
        if hasattr(self.fig, attr):
            return getattr(self.fig, attr)

        else:
            raise AttributeError(
                "%s has not attribute '%s'" % (self.fig, attr))
