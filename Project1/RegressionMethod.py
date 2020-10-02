import numpy as np
from sklearn.linear_model import Lasso
from RegressionResults import RegressionResults

import enum
# Using enum class create enumerations
class RegressionType(enum.Enum):
   OLS = 0
   Ridge = 1
   Lasso = 2

class RegressionMethod(object):
    def __init__(self, model_type = RegressionType.OLS, alpha = 0.0):
        self.model_type = model_type
        self.alpha = alpha
        self.test_results = RegressionResults()
        self.train_results = RegressionResults()

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
        if self.model_type == RegressionType.OLS:
            self.find_beta_OLS()
        elif self.model_type == RegressionType.Ridge:
            self.find_beta_Ridge()
        elif self.model_type == RegressionType.Lasso:
            self.find_beta_Lasso()
        else:
            print("Something went wrong, no model type is sat")

    
    def find_beta_OLS(self):
        self.beta = np.linalg.pinv(self.X_train.T @ self.X_train) @ self.X_train.T @ self.y_train
    
    def find_beta_Ridge(self):
        U, s, V = np.linalg.svd(self.X_train, full_matrices = False) # goddamn magic

        lambda_inverse = np.diag(1/(s**2 + self.alpha))
        #print("V: ", V.shape, ", Lambda-1: ", lambda_inverse.shape, ", s: ", np.diag(s).shape, ", U.T: ", U.T.shape, ", y_train: ", self.y_train.shape)
        self.beta = V.T @ lambda_inverse @ np.diag(s) @ U.T @ self.y_train

    def find_beta_Lasso(self):
        # TODO: should fit_intercept be false or true?
        clf = Lasso(alpha = self.alpha, fit_intercept=False, normalize=False, max_iter=10000, tol=0.006).fit(self.X_train, self.y_train)
        self.beta = clf.coef_

    def __repr__(self):
        return ', '.join("%s: %s" % item for item in vars(self).items())
