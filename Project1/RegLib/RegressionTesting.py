from RegLib.RegressionMethod import RegressionMethod, RegressionType
from RegLib.SamplingMethod import SamplingMethod
from RegLib.HelperFunctions import create_frankie_data, create_X
from RegLib.BootstrapSampling import BootstrapSampling
#from CrossValidationKFold import CrossValidationKFold

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold,cross_val_score, cross_val_predict, cross_validate
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

    N = 20
    noise_strength = 0.1
    p = 5 

    x, y, z = create_frankie_data(seed, N, noise_strength)
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

def test_mean_and_std_of_scaled_data(X, z):
    print("Testing mean and std of scaled data")

    X_train, X_test, Y_train, Y_test = train_test_split(X, z, test_size = 0.2)
    train_data_scaled, test_data_scaled = SamplingMethod.scale_standard(X_train, X_test)

    assert(np.isclose(np.mean(train_data_scaled), 0, atol = 1e-15, equal_nan=True), np.mean(train_data_scaled))
    assert(np.isclose(np.std(train_data_scaled), 1, atol = 1e-15, equal_nan=True), np.std(train_data_scaled))
    assert(np.isclose(np.mean(test_data_scaled), 0, atol = 1e-15, equal_nan=True), np.mean(test_data_scaled))
    assert(np.isclose(np.std(test_data_scaled), 1, atol = 1e-15, equal_nan=True), np.std(test_data_scaled))

def test_Standard_Scaler(X, z):
    print("Testing Standard Scaler")
    X_train, X_test, Y_train, Y_test = train_test_split(X, z, test_size = 0.2)
    scaler = StandardScaler()
    scaler.fit(X_train)
    skl_X_train = scaler.transform(X_train)
    skl_X_test = scaler.transform(X_test)

    sampling = SamplingMethod()
    sampling.split_and_scale_train_test(X, z)

    diff = np.abs(np.abs(np.mean(sampling.X_train)) - np.abs(np.mean(skl_X_train)))
    assert_msg = "\nDifference between means " + str(diff) + " should be less than " + str(error_tolerance) + ".Sampling: " + str(np.mean(sampling.X_train)) + " model: " + str(np.mean(skl_X_train))
    assert (diff < error_tolerance), assert_msg

def test_r2_with_sklearn(X, z):
    print("Testing r2 compared to sklearn r2 method")

    sampling = SamplingMethod().train_and_test(X, z)

    sklearn_score_on_red_model = r2_score(sampling.y_test, sampling.model.get_y_pred(sampling.X_test))
    diff = np.abs(np.abs(sklearn_score_on_red_model) - np.abs(sampling.r2))
    assert_msg = "\nDifference between r2 scores methods" + str(diff) + " should be less than " + str(error_tolerance)
    assert diff < error_tolerance, assert_msg

def test_mse_with_sklearn(X, z):
    print("Testing MSE compared to sklearn MSE method")

    sampling = SamplingMethod().train_and_test(X, z)

    sklearn_mse = mean_squared_error(sampling.y_test, sampling.model.get_y_pred(sampling.X_test))
    diff = np.abs(np.abs(sklearn_mse) - np.abs(sampling.mse))
    assert_msg = "\nDifference between MSE scores methods" + str(diff) + " should be less than " + str(error_tolerance) + ".Sampling: " + str(sampling.mse) + " model: " + str(sklearn_mse)
    assert diff < error_tolerance, assert_msg

def test_OLS_with_sklearn(X, z):
    print("Testing OLS compared to sklearn")

    sampling = SamplingMethod().train_and_test(X, z, model_type = RegressionType.OLS)

    # Use the same scaled data, to test just the OLS method
    lin_model = LinearRegression(fit_intercept=False).fit(sampling.X_train, sampling.y_train)
    y_test_predict = lin_model.predict(sampling.X_test)
    sklearn_score = r2_score(sampling.y_test, y_test_predict)    

    diff = np.abs(np.abs(sklearn_score) - np.abs(sampling.r2))
    assert_msg = "\nDifference between r2 scores " + str(diff) + " should be less than " + str(error_tolerance)
    assert diff < error_tolerance, assert_msg

def test_OLS_train_MSE_with_sklearn(X, z):
    print("Testing OLS compared to sklearn")

    sampling = SamplingMethod().train_and_test(X, z, model_type = RegressionType.OLS)
    sampling_score = sampling.test_model(sampling.model, sampling.X_train, sampling.y_train).mse

    # Use the same scaled data, to test just the OLS method
    lin_model = LinearRegression(fit_intercept=False).fit(sampling.X_train, sampling.y_train)
    sklearn_y_train_predict = lin_model.predict(sampling.X_train)
    sklearn_score = mean_squared_error(sampling.y_train, sklearn_y_train_predict)    

    diff = np.abs(np.abs(sklearn_score) - np.abs(sampling_score))
    assert_msg = "\nDifference between r2 scores " + str(diff) + " should be less than " + str(error_tolerance)+ ".Sampling: " + str(sampling_score) + " model: " + str(sklearn_score)
    assert diff < error_tolerance, assert_msg

def test_ridge_with_sklearn(X, z):
    print("Testing Ridge compared to sklearn")

    ridge_lambda = 1.0
    sampling = SamplingMethod().train_and_test(X, z, model_type = RegressionType.Ridge, alpha = ridge_lambda)

    lin_model = Ridge(alpha = ridge_lambda, fit_intercept=False).fit(sampling.X_train, sampling.y_train)
    y_test_predict = lin_model.predict(sampling.X_test)
    sklearn_score = r2_score(sampling.y_test, y_test_predict)

    diff = np.abs(np.abs(sklearn_score) - np.abs(sampling.r2))
    assert_msg = "\nDifference between r2 scores " + str(diff) + " should be less than " + str(error_tolerance) + ", sklearn: " + str(sklearn_score) + ", our model: " + str(sampling.r2)
    assert diff < error_tolerance, assert_msg

def test_bootstrap_sampling(X, z):
    print("Testing Bootstrap Sampling")
    
    boot = BootstrapSampling().train_and_test(X, z)
    print("r2 score: ", boot.r2)


if __name__ == "__main__":
    test_true()
    test_false()
    X, z = create_test_data()
    #test_Standard_Scaler()
    # test_r2_with_sklearn(X, z)
    # test_ridge_with_sklearn(X, z)
    # test_OLS_with_sklearn(X, z)
    # test_bootstrap_sampling(X, z)
    #test_mse_with_sklearn(X, z)
    test_OLS_train_MSE_with_sklearn(X, z)

    print("Everything passed")
