#!/usr/bin/env python3
import argparse
import os
import pickle

import numpy as np
from matplotlib import pyplot as plt

import Nz_Fitting

parser = argparse.ArgumentParser(
    description="Test the multi-bin joint fitting procedure.")
parser.add_argument(
    "-n", "--n-samples", type=int, default=100,
    help="number of data samples to generate for covariance estimation "
         "(default: %(default)s)")


if __name__ == "__main__":
    args = parser.parse_args()
    wdir = os.path.dirname(__file__)
    zbins = [
        "0.101z0.301", "0.301z0.501", "0.501z0.701",
        "0.701z0.901", "0.901z1.201"]

    # set up a model with 13 components with linear amplitudes for each
    # tomographic bin
    n_comp = 13
    zmin, zmax = 0.07, 1.41  # redshift range of mocks
    bin_models = [
        Nz_Fitting.GaussianComb(n_comp, zmin, (zmax - zmin) / n_comp)
        for i in range(len(zbins))]

    # load the bin weights and initialize the joint fit model
    with open(os.path.join(wdir, "bin_weights.pickle"), "rb") as f:
        weight_dict = pickle.load(f)
    bin_weights = [weight_dict[zbin] for zbin in zbins]
    joint_model = Nz_Fitting.BinnedRedshiftModel(bin_models, bin_weights)

    # load the tomographic bin test files
    bin_data = []
    for zbin in zbins:
        fpath = os.path.join(wdir, "crosscorr_%s.yaw" % zbin)
        bin_data.append(Nz_Fitting.RedshiftData(*np.loadtxt(fpath).T))
    # load the full sample
    fpath = os.path.join(wdir, "crosscorr_%s.yaw" % "0.101z1.201")
    full_data = Nz_Fitting.RedshiftData(*np.loadtxt(fpath).T)
    # combine data into a multi-bin containter
    joint_data = Nz_Fitting.BinnedRedshiftData(bin_data, full_data)

    # fit the joint model to the joint data
    print("#### joint fit ####")
    opt = Nz_Fitting.CurveFit(joint_data, joint_model)
    bestfit = opt.optimize(n_samples=args.n_samples)
    print("best fit with chi²/dof = %.3f" % opt.chisquareReduced(bestfit))
    # estimate the mean redshift and it's uncertainty
    zmeans = joint_model.mean(bestfit)
    zmeans_err = joint_model.meanError(
        bestfit, percentile=95.0, symmetric=False)  # is -nsigma, + nsigma
    stat_iter = zip([*zbins, "full sample"], zmeans, zmeans_err)
    for zbin, zmean, zmean_err in stat_iter:
        print("#### bin %s ####" % zbin)
        print("mean redshift = %.3f %+.3f %+.3f" % (zmean, *zmean_err))

    # plot the parameter covariance
    bestfit.plotCorr()
    plt.show()
    plt.close()

    # plot the data and best-fit model
    fig = joint_data.plot()
    joint_model.plot(bestfit, fig=fig)
    for ax in np.array(fig.axes).flatten():
        ax.tick_params(
            "both", direction="in",
            bottom=True, top=True, left=True, right=True)
        ax.grid(alpha=0.25)
    fig.text(0.5, 0.01, r"$z$", ha="center", fontsize=13)
    fig.text(
        0.01, 0.5, r"$p(z)$", va="center", rotation="vertical", fontsize=13)
    fig.tight_layout(h_pad=0.0, w_pad=0.0)
    fig.subplots_adjust(left=0.06, bottom=0.06, hspace=0.0, wspace=0.0)
    plt.show()
    plt.close()
