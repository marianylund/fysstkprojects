import numpy as np
from SamplingData import SamplingData

class BootstrapSampling(SamplingData):

    def __init__(self, X, y, model, trials = 10, sample_count = 100):
        super(BootstrapSampling, self).__init__(X, y, model)
        self.trials = trials
        self.sample_count = sample_count
    
    def fit(self):
        self.split_and_scale_train_test()
        self.train_test_bootstrap()

    def train_test_bootstrap(self):
        r2_error_avg = np.zeros(self.trials); mse_error_avg = np.zeros(self.trials); variance_beta_avg = np.zeros(self.trials); bias_avg = np.zeros(self.trials); 
        original_X_train_data = self.model.X_train
        original_y_train_data = self.model.y_train
        for sample in range(self.trials):
            self.model.X_train, self.model.y_train = self.resample(original_X_train_data, original_y_train_data)
            self.model.find_beta()
            self.model.find_optimal_beta()
            self.model.test_model()          
            r2_error_avg[sample] = self.model.r2; mse_error_avg[sample] = self.model.mse; variance_beta_avg[sample] = self.model.variance_beta; bias_avg[sample] = self.model.bias

        self.model.X_train = original_X_train_data
        self.model.y_train = original_y_train_data
        self.model.r2 = np.mean(r2_error_avg); self.model.mse = np.mean(mse_error_avg); self.model.variance_beta = np.mean(variance_beta_avg); self.model.bias = np.mean(bias_avg)

    def resample(self, X_train_data, y_train_data):
        assert(X_train_data.shape[0] == y_train_data.shape[0], "X_train and y_train are not the same length")
        indices = [x for x in range(len(X_train_data))]
        sampled_indices = np.random.choice(indices, self.sample_count, replace=True)
        return X_train_data[sampled_indices], y_train_data[sampled_indices]