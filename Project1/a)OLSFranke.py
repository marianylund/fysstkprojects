import numpy as np
from random import random, seed

from RegLib.SamplingMethod import SamplingMethod
from RegLib.RegressionMethod import RegressionType
from RegLib.HelperFunctions import confidence_interval, create_frankie_data, create_X
from PROJECT_SETUP import SEED

np.random.seed(SEED)

N = 10
noise = 0.0
p = 2

x, y, z = create_frankie_data(SEED, N = N, noise_strength=noise)
X = create_X(x, y, n = p)
perm_index = np.random.permutation(len(z))
sampling = SamplingMethod().train_and_test(X, z, perm_index = perm_index, model_type = RegressionType.OLS)

confidence_interval(X, z, sampling.model.beta, noise, N, save_fig = False)