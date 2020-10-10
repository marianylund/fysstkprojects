import numpy as np
from random import random, seed

from RegLib.SamplingMethod import SamplingMethod
from RegLib.RegressionMethod import RegressionType
from RegLib.BootstrapSampling import BootstrapSampling
from RegLib.HelperFunctions import create_frankie_data, create_X, plot_bias_variance_analysis, plot_values_with_info, progressBar
from PROJECT_SETUP import SEED, SAVE_FIG

np.random.seed(SEED)

# Part b): Bias-variance trade-off and resamplng techniques

# Create Figure similar to Fig 2.11 of Hastie to show overfitting
N = 40
noise = 0.3
p = 15

x, y, z = create_frankie_data(SEED, N = N, noise_strength=noise)
perm_index = np.random.permutation(len(z))

polydegree = np.zeros(p); mse_train = np.zeros(p); mse_test = np.zeros(p); r2_train = np.zeros(p); r2_test = np.zeros(p); 

for degree in range(p):
    progressBar(degree + 1, p)
    polydegree[degree] = degree + 1

    X = create_X(x, y, degree, debug = False)
    sampling = SamplingMethod().train_and_test(X, z, perm_index, RegressionType.OLS, shuffle=False, test_size = 0.3)
    mse_test[degree] = sampling.mse
    r2_test[degree] = sampling.r2

    train_sample = sampling.test_model(sampling.model, sampling.X_train, sampling.y_train)
    mse_train[degree] = train_sample.mse
    r2_train[degree] = train_sample.r2

values_to_plot = {
    "Train error": mse_train,
    "Test error": mse_test,
}

info_to_add = {
    "N: ": N,
    "Noise: ": noise
}

plot_values_with_info(polydegree, values_to_plot, title = "b)TestTrainErrorAsModelComplexity", xlabel = "Polynomial Degree", ylabel = "Prediction Error", info_to_add = info_to_add, save_fig=SAVE_FIG)

# Bias-variance trade-off

N = 50
noise = 0.1
p = 5
trials = 10000
sample_count = N
x, y, z = create_frankie_data(SEED, N,  noise_strength=noise)
perm_index = np.random.permutation(len(z))

polydegree = np.zeros(p); r2_boot = np.zeros(p); mse_boot = np.zeros(p); bias_boot = np.zeros(p); var_boot = np.zeros(p)
r2_boot_train = np.zeros(p); mse_boot_train = np.zeros(p); bias_boot_train = np.zeros(p); var_boot_train = np.zeros(p)

for degree in range(p):
    progressBar(degree + 1, p)

    polydegree[degree] = degree + 1

    X = create_X(x, y, degree, debug = False)
    boot = BootstrapSampling(trials, sample_count).train_and_test(X, z, perm_index = perm_index, model_type = RegressionType.OLS)

    r2_boot[degree] = boot.r2; mse_boot[degree] = boot.mse; bias_boot[degree] = boot.bias; var_boot[degree] = boot.var; 
    r2_boot_train[degree] = boot.r2_train; mse_boot_train[degree] = boot.mse_train; bias_boot_train[degree] = boot.bias_train; var_boot_train[degree] = boot.var_train; 


values_to_plot = {
    "Variance": var_boot,
    "MSE": mse_boot,
    "Bias^2": bias_boot,
}

info_to_add = {
    "N: ": N,
    "Noise: ": noise,
    "Trials: ": trials,

}

plot_bias_variance_analysis(polydegree, values_to_plot, title = "b)BiasVarTradeoff", info_to_add = info_to_add, save_fig = True)

values_to_plot = {
    "Train error": mse_boot_train,
    "Test error": mse_boot,
}

plot_values_with_info(polydegree, values_to_plot, title = "b)TestTrainErrorBootstrap", xlabel = "Polynomial Degree", ylabel = "Prediction Error", info_to_add = info_to_add, save_fig=SAVE_FIG)

