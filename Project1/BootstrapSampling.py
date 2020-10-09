import numpy as np
from SamplingMethod import SamplingMethod
from RegressionMethod import RegressionMethod, RegressionType

class BootstrapSampling(SamplingMethod):

    def __init__(self, trials = 10, sample_count = 100):
        self.trials = trials
        self.sample_count = sample_count
    
    def train_and_test(self, X, y, model_type = RegressionType.OLS, alpha = 0.0, test_size  = 0.2, shuffle = False, normalize=True):
        #assert sample_count >= X.shape[1], "Cannot fit matrix with " + str(X.shape[1]) + " deg of freedom, with just " + str(sample_count) + " samples"
        self.split_and_scale_train_test(X, y, test_size, shuffle, normalize)
        self.train_test_bootstrap(self.X_train, self.X_test, self.y_train, self.y_test, model_type = RegressionType.OLS, alpha = 0.0)
        return self

    def train_test_bootstrap(self, X_train, X_test, y_train, y_test, model_type = RegressionType.OLS, alpha = 0.0):
        y_pred = np.empty((y_test.shape[0], self.trials))
        for sample in range(self.trials):
            resampled_X_train, resampled_y_train = self.__resample(X_train, y_train)
            y_pred[:, sample] = RegressionMethod().fit(resampled_X_train, resampled_y_train, model_type, alpha).get_prediction(X_test).ravel()

        self.r2 = self.R2(y_test, y_pred)
        self.mse = self.MSE(y_test, y_pred)
        self.bias = self.get_bias(y_test, y_pred)
        self.var = self.get_variance(y_pred)

        return self

    def __resample(self, X_train_data, y_train_data):
        #assert(X_train_data.shape[0] == y_train_data.shape[0], "X_train and y_train are not the same length")
        indices = [x for x in range(len(X_train_data))]
        sampled_indices = np.random.choice(indices, self.sample_count, replace=True)
        return X_train_data[sampled_indices], y_train_data[sampled_indices]


    @staticmethod
    def get_variance(y_pred):
        return np.mean( np.var(y_pred, axis=1, keepdims=True) )

    @staticmethod
    def R2(y_data, y_pred):
        return 1 - np.sum((y_data - y_pred)**2) / np.sum((y_data - np.mean(y_data))**2)

    @staticmethod
    def MSE(y_data, y_pred):
        return np.mean(np.mean((y_data - y_pred)**2, axis=1, keepdims=True))

    @staticmethod
    def get_bias(y_data, y_pred):
        return np.mean((y_data - np.mean(y_pred, axis=1, keepdims=True))**2)