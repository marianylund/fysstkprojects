import numpy as np
from sklearn.linear_model import Lasso

import enum
# Using enum class create enumerations
class RegressionType(enum.Enum):
   OLS = 0
   Ridge = 1
   Lasso = 2

class RegressionMethod(object):

    def fit(self, X_train, y_train, model_type = RegressionType.OLS, alpha = 0.0):
        self.alpha = alpha

        if model_type == RegressionType.OLS:
            return self.__find_beta_OLS(X_train, y_train)
        elif model_type == RegressionType.Ridge:
            return self.__find_beta_Ridge(X_train, y_train)
        elif model_type == RegressionType.Lasso:
            return self.__find_beta_Lasso(X_train, y_train)
        else:
            print("Something went wrong, no model type is sat")
            return self

    def __find_beta_OLS(self, X_train, y_train):
        #self.beta = np.linalg.pinv(X_train) @ y_train
        self.beta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ y_train
        return self
    
    def __find_beta_Ridge(self, X_train, y_train):
        U, s, V = np.linalg.svd(X_train, full_matrices = False)

        lambda_inverse = np.diag(1/(s**2 + self.alpha))
        #print("V: ", V.shape, ", Lambda-1: ", lambda_inverse.shape, ", s: ", np.diag(s).shape, ", U.T: ", U.T.shape, ", y_train: ", y_train.shape)
        self.beta = V.T @ lambda_inverse @ np.diag(s) @ U.T @ y_train
        return self

    def __find_beta_Lasso(self, X_train, y_train):
        clf = Lasso(alpha = self.alpha, fit_intercept=False, normalize=False, max_iter=10000, tol=0.006).fit(X_train, y_train)
        self.beta = clf.coef_
        return self

    def get_y_pred(self, X_data):
        return X_data @ self.beta

    def __repr__(self):
        return ', '.join("%s: %s" % item for item in vars(self).items())
