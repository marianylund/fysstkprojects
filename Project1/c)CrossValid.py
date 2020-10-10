import numpy as np
from random import random, seed

import matplotlib.pyplot as plt

from RegLib.SamplingMethod import SamplingMethod
from RegLib.RegressionMethod import RegressionType
from RegLib.BootstrapSampling import BootstrapSampling
from RegLib.CrossValidationKFold import CrossValidationKFold
from RegLib.HelperFunctions import create_frankie_data, create_X, plot_bias_variance_analysis, plot_values_with_info, progressBar
from PROJECT_SETUP import SEED, SAVE_FIG

np.random.seed(SEED)

# Part c): Cross-validation as resampling techniques, adding more complexity 

# MSE in Cross validation vs bootstrap

N = 40
noise = 0.3
p = 5

trials = 1000
sample_count = N

kfolds = 5

x, y, z = create_frankie_data(SEED, N,  noise_strength=noise)
perm_index = np.random.permutation(len(z))

polydegree = np.zeros(p); mse_boot = np.zeros(p); mse_kfold = np.zeros(p); mse_boot_train = np.zeros(p); mse_kfold_train = np.zeros(p); 

for degree in range(p):
    progressBar(degree + 1, p)

    polydegree[degree] = degree + 1

    X = create_X(x, y, degree, debug = False)

    crossval = CrossValidationKFold(kfolds).train_and_test(X, z, perm_index = perm_index, model_type = RegressionType.OLS)
    boot = BootstrapSampling(trials, sample_count).train_and_test(X, z, perm_index = perm_index, model_type = RegressionType.OLS)

    mse_kfold[degree] = crossval.mse
    mse_kfold_train[degree] = crossval.mse_train
    mse_boot[degree] = boot.mse
    mse_boot_train[degree] = boot.mse_train

values_to_plot = {
    "Bootstrap Test": mse_boot,
    "Cross-validation Test": mse_kfold,
    "Bootstrap Train": mse_boot_train,
    "Cross-validation Train": mse_kfold_train,
}

info_to_add = {
    "N: ": N,
    "Noise: ": noise,
    "Trials: ": trials,
    "Kfolds: ": kfolds
}

plot_values_with_info(polydegree, values_to_plot, title = "CrossvalidationVSBootstrap", xlabel = "Polynomial Degree", ylabel = "Prediction Error", info_to_add = info_to_add, save_fig=SAVE_FIG)


# Test different kfolds values:

# N = 40
# noise = 0.3
# p = 3

# kfolds = 5

# x, y, z = create_frankie_data(SEED, N,  noise_strength=noise)
# perm_index = np.random.permutation(len(z))


# # Initialize the figure
# plt.style.use('seaborn-darkgrid')
 
# # create a color palette
# palette = plt.get_cmap('Set1')
 
# # multiple line plot
# for fold in range(1, 7):
#     progressBar(fold, 5)
 
#     # Find the right spot on the plot
#     plt.subplot(3, 2, fold)
    
#     polydegree = np.zeros(p); mse_boot = np.zeros(p); mse_kfold = np.zeros(p); mse_boot_train = np.zeros(p); mse_kfold_train = np.zeros(p); 
#     for degree in range(p):

#         polydegree[degree] = degree + 1

#         X = create_X(x, y, degree, debug = False)

#         crossval = CrossValidationKFold(fold + 4).train_and_test(X, z, perm_index = perm_index, model_type = RegressionType.OLS)

#         mse_kfold[degree] = crossval.mse
#         mse_kfold_train[degree] = crossval.mse_train

#     # Plot the lineplot
#     plt.plot(polydegree, mse_kfold, marker='', color=palette(fold), linewidth=1.9, alpha=0.9, label=("Fold: " + str(fold)))
 
#     # Same limits for everybody!
#     plt.xlim(0,10)
#     plt.ylim(1,p)
 
#     # Not ticks everywhere
#     if fold in range(7) :
#         plt.tick_params(labelbottom='off')
#     if fold not in [1,4] :
#         plt.tick_params(labelleft='off')
 
#     # Add title
#     plt.title("Fold: " + str(fold), loc='left', fontsize=12, fontweight=0, color=palette(fold) )
 
# # general title
# plt.suptitle("How the 9 students improved\nthese past few days?", fontsize=13, fontweight=0, color='black', style='italic', y=1.02)
 
# # Axis title
# plt.text(0.5, 0.02, 'Time', ha='center', va='center')
# plt.text(0.06, 0.5, 'Note', ha='center', va='center', rotation='vertical')



# values_to_plot = {
#     "Cross-validation Test": mse_kfold,
#     "Cross-validation Train": mse_kfold_train,
# }

# info_to_add = {
#     "N: ": N,
#     "Noise: ": noise,
#     "Kfolds: ": kfolds
# }

# plot_values_with_info(polydegree, values_to_plot, title = "CrossValidationKfoldValues", xlabel = "Polynomial Degree", ylabel = "Prediction Error", info_to_add = info_to_add, save_fig=SAVE_FIG)
