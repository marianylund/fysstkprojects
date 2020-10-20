import numpy as np
from RegLib.SamplingMethod import SamplingMethod
from RegLib.RegressionMethod import RegressionMethod, RegressionType

class BootstrapSampling(SamplingMethod):

    def __init__(self, trials = 10, sample_count = 100):
        self.trials = trials
        self.sample_count = sample_count
    
    def train_and_test(self, X, y, perm_index = [-1], model_type = RegressionType.OLS, alpha = 0.0, test_size  = 0.2):
        #assert sample_count >= X.shape[1], "Cannot fit matrix with " + str(X.shape[1]) + " deg of freedom, with just " + str(sample_count) + " samples"
        self.split_and_scale_train_test(X, y, perm_index, test_size)
        self.train_test_bootstrap(self.X_train, self.X_test, self.y_train, self.y_test, model_type = RegressionType.OLS, alpha = 0.0)
        return self

    def train_test_bootstrap(self, X_train, X_test, y_train, y_test, model_type = RegressionType.OLS, alpha = 0.0):
        y_pred = np.empty((y_test.shape[0], self.trials))
        y_pred_train = np.empty((y_train.shape[0], self.trials))
        for sample in range(self.trials):
            resampled_X_train, resampled_y_train = self.__resample(X_train, y_train)
            model = RegressionMethod().fit(resampled_X_train, resampled_y_train, model_type, alpha)
            y_pred[:, sample] = model.get_y_pred(X_test).ravel()
            y_pred_train[:, sample] = model.get_y_pred(X_train).ravel()
        
        self.y_pred = y_pred
        self.y_pred_train = y_pred_train

        self.r2 = self.R2(y_test, y_pred)
        self.mse = self.MSE(y_test, y_pred)
        self.bias = self.get_bias(y_test, y_pred)
        self.var = self.get_variance(y_pred)

        self.r2_train = self.R2(y_train, y_pred_train)
        self.mse_train = self.MSE(y_train, y_pred_train)
        self.bias_train = self.get_bias(y_train, y_pred_train)
        self.var_train = self.get_variance(y_pred_train)

        return self

    def to_dict(self, with_prediction = False, with_test_results_on_test_data = True, with_test_results_on_train_data = False):
        
        method_dict = self.model.to_dict()
        method_dict['sampling_method'] = self.__class__.__name__
        method_dict['trials'] = self.trials
        method_dict['sample_count'] = self.sample_count

        if with_prediction:
            method_dict['test_prediction'] =  self.y_pred.tolist()
            method_dict['train_prediction'] =  self.y_pred_train.tolist()
        if with_test_results_on_test_data:
            method_dict['test_r2'] =  self.r2
            method_dict['test_mse'] =  self.mse
            method_dict['test_bias'] =  self.bias
        if with_test_results_on_train_data:
            method_dict['train_r2'] =  self.r2_train
            method_dict['train_mse'] =  self.mse_train
            method_dict['train_bias'] =  self.var_train

        return method_dict

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