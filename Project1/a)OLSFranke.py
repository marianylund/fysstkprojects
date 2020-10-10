import numpy as np
from random import random, seed

from RegLib.SamplingMethod import SamplingMethod
from RegLib.RegressionMethod import RegressionType
from RegLib.HelperFunctions import confidence_interval, create_frankie_data, create_X
from PROJECT_SETUP import SEED, SAVE_FIG

np.random.seed(SEED)

N = 100
noise = 0.3
p = 5

x, y, z = create_frankie_data(SEED, N = N, noise_strength=noise)
X = create_X(x, y, n = p)
perm_index = np.random.permutation(len(z))
sampling = SamplingMethod().train_and_test(X, z, perm_index = perm_index, model_type = RegressionType.OLS)

info_to_add = {
    "N: ": N,
    "Noise: ": noise,
    "Polynomial degree: " : p,
    "MSE: " : sampling.mse,
    "R2: ": sampling.r2
}

confidence_interval(X, z, sampling.model.beta, noise, N, info_to_add = info_to_add, save_fig = SAVE_FIG)