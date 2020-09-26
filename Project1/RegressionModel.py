import numpy as np

class RegressionModel(object):
    def __init__(self, debug = False):
        self.debug = debug

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
        self.beta_optimal = self.X_test @ self.beta
    
    def test_model(self):
        self.r2 = self.R2(self.y_test, self.beta_optimal)
        self.mse = self.MSE(self.y_test, self.beta_optimal)
        self.variance_beta = np.var(self.beta_optimal)
        self.bias = self.get_bias(self.y_test, self.beta_optimal)**2

    def find_beta(self):
        self.beta = np.linalg.pinv(self.X_train.T @ self.X_train) @ self.X_train.T @ self.y_train
    
    def __repr__(self):
        return ', '.join("%s: %s" % item for item in vars(self).items())

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


