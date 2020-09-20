import numpy as np
from sklearn.model_selection import train_test_split

def R2(y_data, y_model):
    return 1 - np.sum((y_data - y_model)**2) / np.sum((y_data - np.mean(y_data))**2)

def MSE(y_data, y_model):
    n = np.size(y_model)
    return np.sum((y_data - y_model)**2)/n

def get_bias(y_data, y_model):
    return np.mean(y_data - np.mean(y_model)**2)

def create_train_test_data(X, y, test_size  = 0.2, shuffle = False, normalize=True, debug=False):
    assert X.shape[0] == y.shape[0], ("X.shape[0] and y.shape[0] needs to be the same length, but: " + str(X.shape[0]) + " != " + str(y.shape[0]))
        
    if debug:
        print("X:\n", X.shape, "\n", X)
        print("y:\n", y.shape, "\n", y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=shuffle)
    X_train_mean = np.mean(X_train, axis = 0)
    X_train_std = np.std(X_train, axis = 0)

    X_train_new = (X_train - X_train_mean)/X_train_std
    X_test_new = (X_test - X_train_mean)/X_train_std

    # Delete the intersect completely or just set to 0
    X_train_new[:, 0] = 1
    X_test_new[:, 0] = 1

    if debug:
        print("Mean: ", X_train_mean, " std: ", X_train_std)
        print("X_train before:\n", X_train, "\nX_train after:\n", X_train_new)

    return X_train_new, X_test_new, y_train, y_test

def find_beta(X_train, y_train, debug = False):
    beta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
    if debug:
        print("Beta: ", beta)
    return beta

def make_prediction_and_test(X_data, beta, y_data, debug_text = ""):
    beta_optimal = X_data @ beta
    r2 = R2(y_data, beta_optimal)
    mse = MSE(y_data, beta_optimal)
    if debug_text != "":
        print(debug_text, "\nBeta optimal: ", beta_optimal, "\nR2(~1): ", r2, "\nMSE(~0): ", mse)
    return beta_optimal, r2, mse

def train_test(X_data, y_data, debug = False):
    beta = find_beta(X_data, y_data, debug)
    beta_optimal, r2_error, mse_error = make_prediction_and_test(X_data, beta, y_data)
    variance_beta = np.var(beta_optimal)
    bias = get_bias(y_data, beta_optimal)
    return beta, beta_optimal, r2_error, mse_error, variance_beta, bias

