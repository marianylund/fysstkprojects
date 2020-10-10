import numpy as np
from random import random, seed
import matplotlib.pyplot as plt

from RegLib.HelperFunctions import plot_3d_graph
from PROJECT_SETUP import TERRAIN_PATH, SAVE_FIG, SEED
from RegLib.SamplingMethod import SamplingMethod
from RegLib.RegressionMethod import RegressionType
from RegLib.BootstrapSampling import BootstrapSampling
from RegLib.CrossValidationKFold import CrossValidationKFold
from RegLib.HelperFunctions import mupltiple_line_plot, plot_bias_variance_analysis, create_terrain_data, create_X, plot_values_with_info, progressBar

np.random.seed(SEED)

# g) OLS, Ridge and Lasso regression with resampling 

#region Alpha values for Lasso and Ridge

# N = 100
# p = 7

# alphas = [0.0, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
# l = len(alphas)

# trials = 1000
# sample_count = N

# kfolds = 5

# x, y, z = create_terrain_data(N, TERRAIN_PATH)
# perm_index = np.random.permutation(len(z))

# mse_boot = np.zeros(l); mse_kfold = np.zeros(l); mse_boot_train = np.zeros(l); mse_kfold_train = np.zeros(l); 
# bias_boot = np.zeros(l); var_boot = np.zeros(l)

# for i in range(len(alphas)):
#     progressBar(i + 1, l)
#     X = create_X(x, y, p, debug = False)

#     crossval = CrossValidationKFold(kfolds).train_and_test(X, z, perm_index = perm_index, model_type = RegressionType.Lasso, alpha = alphas[i])
#     boot = BootstrapSampling(trials, sample_count).train_and_test(X, z, perm_index = perm_index, model_type = RegressionType.Lasso, alpha = alphas[i])

#     mse_kfold[i] = crossval.mse
#     mse_kfold_train[i] = crossval.mse_train
#     mse_boot[i] = boot.mse
#     mse_boot_train[i] = boot.mse_train
#     bias_boot[i] = boot.bias
#     var_boot[i] = boot.var

# values_to_plot = {
#     "Bootstrap Test": mse_boot,
#     "Cross-validation Test": mse_kfold,
#     "Bootstrap Train": mse_boot_train,
#     "Cross-validation Train": mse_kfold_train,
# }

# info_to_add = {
#     "N: ": N,
#     "Trials: ": trials,
#     "Kfolds: ": kfolds, 
#     "Regression": " Lasso",
# }

# plot_values_with_info(alphas, values_to_plot, title = "g)TerrainLassoAlphaBootVsCross", xlabel = "λ values", ylabel = "Prediction Error", info_to_add = info_to_add, xscale = "log", save_fig=SAVE_FIG)


#endregion

#region Bias-variance trade-off

N = 100
p = 10
trials = 10000
sample_count = N
kfolds = 5
alpha = 10
x, y, z = create_terrain_data(N, TERRAIN_PATH)
perm_index = np.random.permutation(len(z))

polydegree = np.zeros(p); r2_boot = np.zeros(p); mse_boot = np.zeros(p); bias_boot = np.zeros(p); var_boot = np.zeros(p)
r2_crossval = np.zeros(p); mse_crossval = np.zeros(p); bias_crossval = np.zeros(p); var_crossval = np.zeros(p)

r2_boot_ridge = np.zeros(p); mse_boot_ridge = np.zeros(p); bias_boot_ridge = np.zeros(p); var_boot_ridge = np.zeros(p)
r2_crossval_ridge = np.zeros(p); mse_crossval_ridge = np.zeros(p); bias_crossval_ridge = np.zeros(p); var_crossval_ridge = np.zeros(p)

r2_boot_lasso = np.zeros(p); mse_boot_lasso = np.zeros(p); bias_boot_lasso = np.zeros(p); var_boot_lasso = np.zeros(p)
r2_crossval_lasso = np.zeros(p); mse_crossval_lasso = np.zeros(p); bias_crossval_lasso = np.zeros(p); var_crossval_lasso = np.zeros(p)

for degree in range(p):
    progressBar(degree + 1, p)

    polydegree[degree] = degree + 1

    X = create_X(x, y, degree, debug = False)
    boot = BootstrapSampling(trials, sample_count).train_and_test(X, z, perm_index = perm_index, model_type = RegressionType.OLS)
    crossval = CrossValidationKFold(kfolds).train_and_test(X, z, perm_index = perm_index, model_type = RegressionType.OLS)

    boot_ridge = BootstrapSampling(trials, sample_count).train_and_test(X, z, perm_index = perm_index, model_type = RegressionType.Ridge, alpha=alpha)
    crossval_ridge = CrossValidationKFold(kfolds).train_and_test(X, z, perm_index = perm_index, model_type = RegressionType.Ridge, alpha=alpha)

    boot_lasso = BootstrapSampling(trials, sample_count).train_and_test(X, z, perm_index = perm_index, model_type = RegressionType.Lasso, alpha=alpha)
    crossval_lasso = CrossValidationKFold(kfolds).train_and_test(X, z, perm_index = perm_index, model_type = RegressionType.Lasso, alpha=alpha)

    r2_boot[degree] = boot.r2; mse_boot[degree] = boot.mse; bias_boot[degree] = boot.bias; var_boot[degree] = boot.var; 
    r2_crossval[degree] = crossval.r2; mse_crossval[degree] = crossval.mse; bias_crossval[degree] = crossval.bias; var_crossval[degree] = crossval.var; 

    r2_boot_ridge[degree] = boot_ridge.r2; mse_boot_ridge[degree] = boot_ridge.mse; bias_boot_ridge[degree] = boot_ridge.bias; var_boot_ridge[degree] = boot_ridge.var; 
    r2_crossval_ridge[degree] = crossval_ridge.r2; mse_crossval_ridge[degree] = crossval_ridge.mse; bias_crossval_ridge[degree] = crossval_ridge.bias; var_crossval_ridge[degree] = crossval_ridge.var; 

    r2_boot_lasso[degree] = boot_lasso.r2; mse_boot_lasso[degree] = boot_lasso.mse; bias_boot_lasso[degree] = boot_lasso.bias; var_boot_lasso[degree] = boot_lasso.var; 
    r2_crossval_lasso[degree] = crossval_lasso.r2; mse_crossval_lasso[degree] = crossval_lasso.mse; bias_crossval_lasso[degree] = crossval_lasso.bias; var_crossval_lasso[degree] = crossval_lasso.var; 


values_to_plot_boot = {
    "Variance": var_boot,
    "MSE": mse_boot,
    "Bias^2": bias_boot,
}

values_to_plot_crossval = {
    "Variance": var_crossval,
    "MSE": mse_crossval,
    "Bias^2": bias_crossval,
}

values_to_plot_boot_ridge = {
    "Variance": var_boot_ridge,
    "MSE": mse_boot_ridge,
    "Bias^2": bias_boot_ridge,
}

values_to_plot_crossval_ridge = {
    "Variance": var_crossval_ridge,
    "MSE": mse_crossval_ridge,
    "Bias^2": bias_crossval_ridge,
}

values_to_plot_boot_lasso = {
    "Variance": var_boot_lasso,
    "MSE": mse_boot_lasso,
    "Bias^2": bias_boot_lasso,
}

values_to_plot_crossval_lasso = {
    "Variance": var_crossval_lasso,
    "MSE": mse_crossval_lasso,
    "Bias^2": bias_crossval_lasso,
}

info_to_add = {
    "N: ": N,
    "Trials: ": trials,
    "Kfolds: ": kfolds,
    "Alpha: ": alpha,
}

values_to_plot = [values_to_plot_boot, values_to_plot_crossval, values_to_plot_boot_ridge, values_to_plot_crossval_ridge, values_to_plot_boot_lasso, values_to_plot_crossval_lasso]
plot_labels = ["OLS Bootstrap", "OLS Cross Validation", "Ridge Bootstrap", "Ridge Cross Validation", "Lasso Bootstrap", "Lasso Cross Validation"]
ylim = [0, 10000]

mupltiple_line_plot(polydegree, values_to_plot, plot_labels, "Terrain Bias Variance Analysis", info_to_add = info_to_add, xlabel = "Polynomial degree", ylabel = "Prediction Error", ylim = ylim, xscale = "linear", save_fig = SAVE_FIG)
#plot_bias_variance_analysis(polydegree, values_to_plot, title = "g)TerrainBiasVarTradeoffOLS", info_to_add = info_to_add, save_fig = SAVE_FIG)

#plot_values_with_info(polydegree, values_to_plot, title = "TestTrainErrorBootstrap", xlabel = "Polynomial Degree", ylabel = "Prediction Error", info_to_add = info_to_add, save_fig=False)


# #endregion
