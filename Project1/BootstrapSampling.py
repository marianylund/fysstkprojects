import numpy as np
from RegressionModel import RegressionModel
from SamplingData import SamplingData
from helper_func import *

class BootstrapSampling(SamplingData):

    def train_test_bootstrap(self, X_train_data, y_train_data, X_test_data, y_test_data, trials, sample_count = 1000, debug = False):
        r2_error_avg = np.zeros(trials); mse_error_avg = np.zeros(trials); variance_beta_avg = np.zeros(trials); bias_avg = np.zeros(trials); 

        for sample in range(trials):
            X_data, y_data = self.resample(X_train_data, y_train_data, sample_count)
            beta = find_beta(X_train_data, y_train_data, debug)            
            beta_train, beta_optimal_train, r2_error_avg[sample], mse_error_avg[sample], variance_beta_avg[sample], bias_avg[sample] = test_data(X_test_data, y_test_data, beta, debug)

        return np.mean(r2_error_avg[sample]), np.mean(mse_error_avg[sample]), np.mean(variance_beta_avg[sample]), np.mean(bias_avg[sample])

    def resample(self, X_train_data, y_train_data, sample_count = 1000):
        assert(X_train_data.shape[0] == y_train_data.shape[0], "X_train and y_train are not the same length")
        indices = [x for x in range(len(X_train_data))]
        sampled_indices = np.random.choice(indices, sample_count, replace=True)
        return X_train_data[sampled_indices], y_train_data[sampled_indices]