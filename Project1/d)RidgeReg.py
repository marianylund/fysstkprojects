import numpy as np
from random import random, seed

import matplotlib.pyplot as plt

from RegLib.SamplingMethod import SamplingMethod
from RegLib.RegressionMethod import RegressionType
from RegLib.BootstrapSampling import BootstrapSampling
from RegLib.CrossValidationKFold import CrossValidationKFold
from RegLib.HelperFunctions import plot_bias_variance_analysis, create_frankie_data, create_X, plot_bias_variance_analysis, plot_values_with_info, progressBar
from PROJECT_SETUP import SEED

np.random.seed(SEED)

# Part d): Ridge Regression on the Franke function with resampling 

# Cross valid and bootstrap for different alphas

N = 40
noise = 0.3
p = 4

alphas = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.0, 1.0, 10.0, 100.0]
l = len(alphas)

trials = 1000
sample_count = N

kfolds = 5

x, y, z = create_frankie_data(SEED, N,  noise_strength=noise)
perm_index = np.random.permutation(len(z))

mse_boot = np.zeros(l); mse_kfold = np.zeros(l); mse_boot_train = np.zeros(l); mse_kfold_train = np.zeros(l); 
bias_boot = np.zeros(l); var_boot = np.zeros(l)

for i in range(len(alphas)):
    progressBar(i + 1, l)

    X = create_X(x, y, p, debug = False)

    crossval = CrossValidationKFold(kfolds).train_and_test(X, z, perm_index = perm_index, model_type = RegressionType.Ridge, alpha = alphas[i])
    boot = BootstrapSampling(trials, sample_count).train_and_test(X, z, perm_index = perm_index, model_type = RegressionType.Ridge, alpha = alphas[i])

    mse_kfold[i] = crossval.mse
    mse_kfold_train[i] = crossval.mse_train
    mse_boot[i] = boot.mse
    mse_boot_train[i] = boot.mse_train
    bias_boot[i] = boot.bias
    var_boot[i] = boot.var

values_to_plot = {
    "Bootstrap Test": mse_boot,
    "Cross-validation Test": mse_kfold,
    #"Bootstrap Train": mse_boot_train,
    #"Cross-validation Train": mse_kfold_train,
}

info_to_add = {
    "N: ": N,
    "Noise: ": noise,
    "Trials: ": trials,
    "Kfolds: ": kfolds, 
    "Regression": " Ridge",
}

#plot_values_with_info(alphas, values_to_plot, title = "RidgeAlphaBootVsCross", xlabel = "λ values", ylabel = "Prediction Error", info_to_add = info_to_add, xscale = "log", save_fig=False, scatter = True)

values_to_plot2 = {
    # "Bootstrap Test": mse_boot,
    #"Cross-validation Test": mse_kfold,
    "Bootstrap Train": mse_boot_train,
    "Cross-validation Train": mse_kfold_train,
}

#plot_values_with_info(alphas, values_to_plot2, title = "RidgeAlphaBootVsCrossTrain", xlabel = "λ values", ylabel = "Prediction Error", info_to_add = info_to_add, xscale = "log", save_fig=False, scatter = True)

# Bias-variance trade-off using bootstrap for different alphas

values_to_plot = {
    "Variance": var_boot,
    "MSE": mse_boot,
    "Bias^2": bias_boot,
}

info_to_add = {
    "N: ": N,
    "Noise: ": noise,
    "Trials: ": trials,
    "Regression": " Ridge",
}

#plot_values_with_info(alphas, values_to_plot, title = "BiasVarTradeoffRidgeAlphas", xlabel = "λ values", ylabel = "Prediction Error", info_to_add = info_to_add, xscale = "log", save_fig=False, scatter = True)

