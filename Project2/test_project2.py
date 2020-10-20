import numpy as np
from random import random, seed

from RegLib.SamplingMethod import SamplingMethod
from RegLib.RegressionMethod import RegressionType
from RegLib.HelperFunctions import confidence_interval, create_frankie_data, create_X
from PROJECT_SETUP import SEED, SAVE_FIG

error_tolerance = 1e-10

def create_test_data():
    N = 20
    noise_strength = 0.1
    p = 5 

    x, y, z = create_frankie_data(SEED, N, noise_strength)
    X = create_X(x, y, p)
    return X, z

def test_saving_model(X, z):
    print("Testing Saving Model")
    
    perm_index = np.random.permutation(len(z))
    sampling = SamplingMethod().train_and_test(X, z, perm_index = perm_index, model_type = RegressionType.OLS)


def test_loading_saved_model(X, z):
    print("Testing Loading Saved Model")
    

if __name__ == "__main__":
    print("Start tests for project 2")
    X, z = create_test_data()
    test_saving_model(X, z)
    test_loading_saved_model(X, z)

    print("All tests have passed")
