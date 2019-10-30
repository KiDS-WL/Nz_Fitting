#!/usr/bin/env python3
import os
import pickle

import numpy as np
from matplotlib import pyplot as plt

import Nz_Fitting

if __name__ == "__main__":

    wdir = os.path.dirname(__file__)
    # load test files with columns (redshift, n(z), n(z) error)
    zbins = [
        "0.101z0.301", "0.301z0.501", "0.501z0.701",
        "0.701z0.901", "0.901z1.201"]
    with open(os.path.join(wdir, "bin_weights.pickle"), "rb") as f:
        weight_dict = pickle.load(f)
    bins = []
    weights = []
    for zbin in zbins:
        fpath = os.path.join(wdir, "crosscorr_%s.yaw" % zbin)
        data = Nz_Fitting.RedshiftData(*np.loadtxt(fpath).T)
        bins.append(data)
        weights.append(weight_dict[zbin])
    fpath = os.path.join(wdir, "crosscorr_%s.yaw" % "0.101z1.201")
    master = Nz_Fitting.RedshiftData(*np.loadtxt(fpath).T)
    # combine data into a multi-bin containter
    data = Nz_Fitting.BinnedRedshiftData(bins, master)
    # set up a model with 13 components with linear amplitudes
    n_comp = 13
    zmin, zmax = 0.07, 1.41  # range of mocks
    models = [
        Nz_Fitting.GaussianComb(n_comp, zmin, (zmax - zmin) / n_comp)
        for i in range(len(bins))]
    model = Nz_Fitting.BinnedRedshiftModel(models, weights)
    # fit the model to the data
    opt = Nz_Fitting.CurveFit(data, model)
    bestfit = opt.optimize(n_samples=100)
    print("best fit with chi²/dof = %.3f" % opt.chisquareReduced(bestfit))
    zmeans = ", ".join("%.3f" % z for z in model.mean(bestfit))
    zmeans_err = ", ".join("%.3f" % z for z in model.meanError(bestfit))
    print("mean redshift = [%s] +- [%s]" % (zmeans, zmeans_err))
    # plot the parameter covariance
    bestfit.plotCorr()
    plt.show()
    # plot the data and best-fit model
    fig = data.plot()
    model.plot(bestfit, fig=fig)
    plt.show()
