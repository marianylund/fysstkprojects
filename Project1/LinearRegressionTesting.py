from RegressionModel import RegressionModel
from SamplingData import SamplingData
from helper_func import *

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np
from random import random, seed

error_tolerance = 1e-14

def test_true():
    assert True, "Should be True"

def test_false():
    assert not False, "Should Fail"

def create_test_data():
    seed = 3155
    np.random.seed(seed)

    N = 20
    noise_strength = 0.1
    p = 2 

    x, y = create_mesh(N, random_mesh = True, seed = seed)
    z_franke = FrankeFunction(x, y, noise_strength)
    z = np.ravel(z_franke)
    X = create_X(x, y, p)
    return X, z

def create_simple_test_data():
    seed = 3155
    np.random.seed(seed)

    x = np.random.rand(100)
    y = 2.0+5*x*x+0.1*np.random.randn(100)
    p = 3
    X = np.zeros((len(x), p))
    X[:, 0] = 1.0
    X[:, 1] = x
    X[:, 2] = x*x

    return X, y

def test_mean_and_std_of_scaled_data():
    print("Testing mean and std of scaled data")

    X, z = create_test_data()
    X_train, X_test, Y_train, Y_test = train_test_split(X, z, test_size = 0.2)
    train_data_scaled, test_data_scaled = SamplingData.scale_standard(X_train, X_test)

    assert(np.isclose(np.mean(train_data_scaled), 0, atol = 1e-15, equal_nan=True), np.mean(train_data_scaled))
    assert(np.isclose(np.std(train_data_scaled), 1, atol = 1e-15, equal_nan=True), np.std(train_data_scaled))
    assert(np.isclose(np.mean(test_data_scaled), 0, atol = 1e-15, equal_nan=True), np.mean(test_data_scaled))
    assert(np.isclose(np.std(test_data_scaled), 1, atol = 1e-15, equal_nan=True), np.std(test_data_scaled))

def test_Standard_Scaler():
    print("Testing Standard Scaler")
    X, z = create_test_data()
    X_train, X_test, Y_train, Y_test = train_test_split(X, z, test_size = 0.2)
    scaler = StandardScaler()
    scaler.fit(X_train)
    skl_X_train = scaler.transform(X_train)
    skl_X_test = scaler.transform(X_test)

    model = RegressionModel()
    sampling = SamplingData(X, z, model)
    sampling.split_and_scale_train_test()

    diff = np.abs(np.abs(np.mean(sampling.model.X_train)) - np.abs(np.mean(skl_X_train)))
    assert_msg = "\nDifference between means " + str(diff) + " should be less than " + str(error_tolerance) + ".Sampling: " + str(np.mean(sampling.model.X_train)) + " model: " + str(np.mean(skl_X_train))
    assert (diff < error_tolerance), assert_msg

def test_OLS_with_sklearn():
    X, z = create_test_data()
    X_train, X_test, Y_train, Y_test = train_test_split(X, z, test_size = 0.2)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    lin_model = LinearRegression(fit_intercept=True).fit(X_train, Y_train)
    y_test_predict = lin_model.predict(X_test)
    sklearn_score = r2_score(Y_test, y_test_predict)

    model = RegressionModel()
    sampling = SamplingData(X, z, model)
    sampling.fit()

    sklearn_score_on_red_model = r2_score(Y_test, sampling.model.beta_optimal)
    print("Same r2 func: ", sklearn_score, sklearn_score_on_red_model, " our error: ", sampling.model.r2)
    diff = np.abs(np.abs(sklearn_score) - np.abs(sampling.model.r2))
    assert_msg = "\nDifference between r2 scores " + str(diff) + " should be less than " + str(error_tolerance)
    assert diff < error_tolerance, assert_msg

if __name__ == "__main__":
    test_true()
    test_false()
    #test_Standard_Scaler()
    test_OLS_with_sklearn()

    print("Everything passed")
