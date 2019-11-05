#!/usr/bin/env python3
import os

import numpy as np
from matplotlib import pyplot as plt

import Nz_Fitting

if __name__ == "__main__":

    ###############################################
    #                                             #
    #  Test the single-bin fitting procedure      #
    #  usage: test_multi_bin.py [n_samples=1000]  #
    #                                             #
    ###############################################
    import sys

    if len(sys.argv) > 1:
        n_samples = int(sys.argv[1])
    else:
        n_samples = 1000

    wdir = os.path.dirname(__file__)
    # path to test file with columns (redshift, n(z), n(z) error)
    fpath = os.path.join(wdir, "crosscorr_0.101z1.201.yaw")
    data = Nz_Fitting.RedshiftData(*np.loadtxt(fpath).T)
    # path to data covariance matrix (only for resampling)
    # fpath = ...
    # data.setCovariance(np.loadtxt(fpath))

    # set up a model with 13 components with linear amplitudes
    n_comp = 13
    zmin, zmax = 0.07, 1.41  # range of mocks
    model = Nz_Fitting.GaussianComb(n_comp, zmin, (zmax - zmin) / n_comp)
    # fit the model to the data
    opt = Nz_Fitting.CurveFit(data, model)
    bestfit = opt.optimize(n_samples=n_samples)
    print("best fit with chi²/dof = %.3f" % opt.chisquareReduced(bestfit))
    zmean = model.mean(bestfit)
    zmean_err = model.meanError(bestfit)
    print("mean redshift = %.3f +- %.3f" % (zmean, zmean_err))
    # plot the parameter covariance
    bestfit.plotSamples()
    plt.show()
    bestfit.plotCorr()
    plt.show()
    # plot the data and best-fit model
    data.plot()
    model.plot(bestfit)
    plt.show()
