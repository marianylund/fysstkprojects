from RegressionMethod import RegressionMethod, RegressionType
from RegressionResults import RegressionResults
from SamplingMethod import SamplingMethod
from HelperFunctions import create_frankie_data, create_X
from BootstrapSampling import BootstrapSampling
from CrossValidationKFold import CrossValidationKFold

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
    p = 2 

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

def test_mean_and_std_of_scaled_data():
    print("Testing mean and std of scaled data")

    X, z = create_test_data()
    X_train, X_test, Y_train, Y_test = train_test_split(X, z, test_size = 0.2)
    train_data_scaled, test_data_scaled = SamplingMethod.scale_standard(X_train, X_test)

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

    model = RegressionMethod()
    sampling = SamplingMethod(X, z, model)
    sampling.split_and_scale_train_test()

    diff = np.abs(np.abs(np.mean(sampling.model.X_train)) - np.abs(np.mean(skl_X_train)))
    assert_msg = "\nDifference between means " + str(diff) + " should be less than " + str(error_tolerance) + ".Sampling: " + str(np.mean(sampling.model.X_train)) + " model: " + str(np.mean(skl_X_train))
    assert (diff < error_tolerance), assert_msg

def test_r2_with_sklearn():
    print("Testing r2 compared to sklearn r2 method")

    X, z = create_test_data()

    model = RegressionMethod()
    sampling = SamplingMethod(X, z, model)
    sampling.fit()

    sampling_model_r2 = sampling.model.test_results.r2
    sklearn_score_on_red_model = r2_score(sampling.model.y_test, sampling.model.test_results.beta_optimal)
    diff = np.abs(np.abs(sklearn_score_on_red_model) - np.abs(sampling_model_r2))
    assert_msg = "\nDifference between r2 scores methods" + str(diff) + " should be less than " + str(error_tolerance)
    assert diff < error_tolerance, assert_msg

def test_OLS_with_sklearn():
    print("Testing OLS compared to sklearn")

    X, z = create_test_data()

    model = RegressionMethod()
    sampling = SamplingMethod(X, z, model)
    sampling.fit()

    # Use the same scaled data, to test just the OLS method
    lin_model = LinearRegression(fit_intercept=False).fit(sampling.model.X_train, sampling.model.y_train)
    y_test_predict = lin_model.predict(sampling.model.X_test)
    sklearn_score = r2_score(sampling.model.y_test, y_test_predict)    

    sampling_model_r2 = sampling.model.test_results.r2

    diff = np.abs(np.abs(sklearn_score) - np.abs(sampling_model_r2))
    assert_msg = "\nDifference between r2 scores " + str(diff) + " should be less than " + str(error_tolerance)
    assert diff < error_tolerance, assert_msg

def test_ridge_with_sklearn():
    print("Testing Ridge compared to sklearn")

    ridge_lambda = 1.0
    X, z = create_test_data()

    model = RegressionMethod(model_type=RegressionType.Ridge, alpha = ridge_lambda)
    sampling = SamplingMethod(X, z, model)
    sampling.fit()
    sampling_model_r2 = sampling.model.test_results.r2

    lin_model = Ridge(alpha = ridge_lambda, fit_intercept=False).fit(sampling.model.X_train, sampling.model.y_train)
    y_test_predict = lin_model.predict(sampling.model.X_test)
    sklearn_score = r2_score(sampling.model.y_test, y_test_predict)

    diff = np.abs(np.abs(sklearn_score) - np.abs(sampling_model_r2))
    assert_msg = "\nDifference between r2 scores " + str(diff) + " should be less than " + str(error_tolerance) + ", sklearn: " + str(sklearn_score) + ", our model: " + str(sampling_model_r2)
    assert diff < error_tolerance, assert_msg

def test_bootstrap_sampling():
    print("Testing Bootstrap Sampling")
    X, z = create_test_data()
    boot = BootstrapSampling(X, z, RegressionMethod())
    boot.fit()
    print("r2 score: ", boot.model.r2)

def test_k_fold_with_sklearn():
    k = 5
    X, z = create_test_data()

    X_train, X_test, y_train, y_test = train_test_split(X, z, test_size=0.2, shuffle=False)
    lm = LinearRegression()
    model = lm.fit(X_train, y_train)

    kfold = KFold(n_splits = k, shuffle = False)
    prediction_scores = cross_validate(model, X, z, cv=kfold, scoring=('neg_mean_squared_error'), return_train_score=True)
    print("Cross-validated train_score per fold from sklearn:", np.mean(-prediction_scores['train_score']))
    print("Cross-validated test_score per fold from sklearn:", np.mean(-prediction_scores['test_score']))

    kfold_sampling = CrossValidationKFold(X, z, RegressionMethod(), k)
    kfold_sampling.fit()

    print("Error score(mse): ", kfold_sampling.model.mse)

if __name__ == "__main__":
    test_true()
    test_false()
    #test_Standard_Scaler()
    #test_r2_with_sklearn()
    test_ridge_with_sklearn()
    #test_OLS_with_sklearn()
    #test_bootstrap_sampling()
    #test_k_fold_with_sklearn()

    print("Everything passed")
