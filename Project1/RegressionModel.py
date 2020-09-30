import numpy as np
from sklearn.linear_model import Lasso

import enum
# Using enum class create enumerations
class ModelType(enum.Enum):
   OLS = 0
   Ridge = 1
   Lasso = 2

class RegressionModel(object):
    def __init__(self, model_type = ModelType.OLS, alpha = 0.0):
        self.model_type = model_type
        self.alpha = alpha
        self.test_results = ModelResults()
        self.train_results = ModelResults()

    def train_and_test(self):
        self.find_beta()
        self.find_optimal_beta()
        self.test_model()

    def set_data(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def find_optimal_beta(self):
        # TODO: prediction, not beta optimal
        self.test_results.beta_optimal = self.X_test @ self.beta
        self.train_results.beta_optimal = self.X_train @ self.beta
    
    def test_model(self):
        self.test_results.test_data(self.y_test)
        self.train_results.test_data(self.y_train)

    def find_beta(self):
        if self.model_type == ModelType.OLS:
            self.find_beta_OLS()
        elif self.model_type == ModelType.Ridge:
            self.find_beta_Ridge()
        elif self.model_type == ModelType.Lasso:
            self.find_beta_Lasso()
        else:
            print("Something went wrong, no model type is sat")

    
    def find_beta_OLS(self):
        self.beta = np.linalg.pinv(self.X_train.T @ self.X_train) @ self.X_train.T @ self.y_train
    
    def find_beta_Ridge(self):
        U, s, V = np.linalg.svd(self.X_train, full_matrices = False) # goddamn magic

        lambda_inverse = np.diag(1/(s**2 + self.alpha))
        #print("V: ", V.shape, ", Lambda-1: ", lambda_inverse.shape, ", s: ", np.diag(s).shape, ", U.T: ", U.T.shape, ", y_train: ", self.y_train.shape)
        self.beta = V @ lambda_inverse @ np.diag(s) @ U.T @ self.y_train

    def find_beta_Lasso(self):
        # TODO: should fit_intercept be false or true?
        clf = Lasso(alpha = self.alpha, fit_intercept=False, normalize=False, max_iter=10000, tol=0.006).fit(self.X_train, self.y_train)
        self.beta = clf.coef_

    def __repr__(self):
        return ', '.join("%s: %s" % item for item in vars(self).items())

class ModelResults():
    def __init__(self, N = 0):
        self.r2 = None
        self.mse = None
        self.variance_beta = None
        self.bias = None
        self.beta_optimal = None

        if(N > 0):
            self.r2_avg = np.zeros(N)
            self.mse_avg = np.zeros(N)
            self.variance_beta_avg = np.zeros(N)
            self.bias_avg = np.zeros(N)
    
    def set_results(self, data, index):
        self.r2_avg[index] = data.r2
        self.mse_avg[index] = data.mse
        self.variance_beta_avg[index] = data.variance_beta
        self.bias_avg[index] = data.bias

    def average_out_results(self):
        self.r2 = np.mean(self.r2_avg)
        self.mse = np.mean(self.mse_avg)
        self.variance_beta = np.mean(self.variance_beta_avg)
        self.bias = np.mean(self.bias_avg)
    
    def set_up_avg(self, N):
        self.r2_avg = np.zeros(N)
        self.mse_avg = np.zeros(N)
        self.variance_beta_avg = np.zeros(N)
        self.bias_avg = np.zeros(N)

    def test_data(self, y_data, index = -1):
        if index == -1:
            self.r2 = self.R2(y_data, self.beta_optimal)
            self.mse = self.MSE(y_data, self.beta_optimal)
            self.variance_beta = np.var(self.beta_optimal)
            self.bias = self.get_bias(y_data, self.beta_optimal)**2
            #print('{} >= {} + {} = {}'.format(self.mse, self.bias, self.variance_beta, self.bias+self.variance_beta))
        else:
            self.r2_avg[index] = self.R2(y_data, self.beta_optimal)
            self.mse_avg[index] = self.MSE(y_data, self.beta_optimal)
            self.variance_beta_avg[index] = np.var(self.beta_optimal)
            self.bias_avg[index] = self.get_bias(y_data, self.beta_optimal)**2
            #print('{} >= {} + {} = {}'.format(self.mse_avg[index], self.bias_avg[index], self.variance_beta_avg[index], self.bias_avg[index]+self.variance_beta_avg[index]))


    def get(self):
        return self.r2, self.mse, self.variance_beta, self.bias
    
    def get_avg(self):
        return self.r2_avg, self.mse_avg, self.variance_beta_avg, self.bias_avg
    
    @staticmethod
    def R2(y_data, y_model):
        return 1 - np.sum((y_data - y_model)**2) / np.sum((y_data - np.mean(y_data))**2)

    @staticmethod
    def MSE(y_data, y_model):
        n = np.size(y_model)
        return np.sum((y_data - y_model)**2)/n

    @staticmethod
    def get_bias(y_data, y_model):
        return np.mean(y_data - np.mean(y_model)**2)
