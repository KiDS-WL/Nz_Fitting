#!/usr/bin/env python3
import argparse
import os
import sys

import numpy as np
from matplotlib import pyplot as plt

import Nz_Fitting

parser = argparse.ArgumentParser(
    description="Test the single-bin fitting procedure.")
parser.add_argument(
    "-n", "--n-samples", type=int, default=1000,
    help="number of data samples to generate for covariance estimation "
         "(default: %(default)s)")
parser.add_argument(
    "-c", "--cov", action="store_true",
    help="use the existing covariance matrices")


if __name__ == "__main__":
    args = parser.parse_args()
    wdir = os.path.dirname(__file__)
    zbins = [
        "0.101z0.301", "0.301z0.501", "0.501z0.701",
        "0.701z0.901", "0.901z1.201"]

    # set up a model with 13 components with linear amplitudes
    n_comp = 13
    zmin, zmax = 0.07, 1.41  # redshift range of mocks
    model = Nz_Fitting.GaussianComb(n_comp, zmin, (zmax - zmin) / n_comp)

    # fit the model to each tomographic bin
    for zbin in zbins:
        print("#### bin %s ####" % zbin)
        fpath = os.path.join(wdir, "crosscorr_%s.yaw" % zbin)
        data = Nz_Fitting.RedshiftData(*np.loadtxt(fpath).T)
        if args.cov:
            fpath = os.path.join(wdir, "crosscorr_%s.cov" % zbin)
            data.setCovariance(np.loadtxt(fpath))
        # fit the model to the data
        opt = Nz_Fitting.CurveFit(data, model)
        bestfit = opt.optimize(n_samples=args.n_samples)
        print("best fit with chi²/dof = %.3f" % opt.chisquareReduced(bestfit))
        # estimate the mean redshift and it's uncertainty
        zmean = model.mean(bestfit)
        zmean_err = model.meanError(
            bestfit, percentile=95.0, symmetric=False)  # is -nsigma, + nsigma
        print("mean redshift = %.3f %+.3f %+.3f" % (zmean, *zmean_err))

    # fit the model to the full data sample
    print("#### full sample ####")
    fpath = os.path.join(wdir, "crosscorr_0.101z1.201.yaw")
    data = Nz_Fitting.RedshiftData(*np.loadtxt(fpath).T)
    if args.cov:
        fpath = os.path.join(wdir, "crosscorr_%s.cov" % zbin)
        data.setCovariance(np.loadtxt(fpath))
    # fit the model to the data
    opt = Nz_Fitting.CurveFit(data, model)
    bestfit = opt.optimize(n_samples=args.n_samples)
    print("best fit with chi²/dof = %.3f" % opt.chisquareReduced(bestfit))
    # estimate the mean redshift and it's uncertainty
    zmean = model.mean(bestfit)
    zmean_err = model.meanError(
        bestfit, percentile=95.0, symmetric=False)  # is -nsigma, + nsigma
    print("mean redshift = %.3f %+.3f %+.3f" % (zmean, *zmean_err))

    # plot the parameter covariance
    bestfit.plotSamples()
    plt.show()
    plt.close()
    bestfit.plotCorr()
    plt.show()
    plt.close()

    # plot the data and best-fit model
    data.plot()
    model.plot(bestfit)
    plt.xlabel(r"$z$", fontsize=13)
    plt.ylabel(r"$p(z)$", fontsize=13)
    plt.grid(alpha=0.25)
    plt.show()
    plt.close()
