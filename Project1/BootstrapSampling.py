import numpy as np
from SamplingData import SamplingData
from RegressionModel import ModelResults

class BootstrapSampling(SamplingData):

    def __init__(self, X, y, model, trials = 10, sample_count = 100):
        super(BootstrapSampling, self).__init__(X, y, model)
        self.trials = trials
        assert sample_count >= X.shape[1], "Cannot fit matrix with " + str(X.shape[1]) + " deg of freedom, with just " + str(sample_count) + " samples"
        self.sample_count = sample_count
    
    def fit(self):
        self.split_and_scale_train_test()
        return self.train_test_bootstrap()

    def train_test_bootstrap(self):
        bootstrap_train_results = ModelResults(self.trials)
        bootstrap_test_results = ModelResults(self.trials)

        original_X_train_data = self.model.X_train
        original_y_train_data = self.model.y_train

        for sample in range(self.trials):
            self.model.X_train, self.model.y_train = self.resample(original_X_train_data, original_y_train_data)
            self.model.find_beta()
            self.model.find_optimal_beta()
            self.model.test_model()

            bootstrap_train_results.set_results(self.model.train_results, sample)
            bootstrap_test_results.set_results(self.model.test_results, sample)

        bootstrap_train_results.average_out_results()
        bootstrap_test_results.average_out_results()

        self.model.X_train = original_X_train_data
        self.model.y_train = original_y_train_data
        return bootstrap_train_results, bootstrap_test_results

    def resample(self, X_train_data, y_train_data):
        assert(X_train_data.shape[0] == y_train_data.shape[0], "X_train and y_train are not the same length")
        indices = [x for x in range(len(X_train_data))]
        sampled_indices = np.random.choice(indices, self.sample_count, replace=True)
        return X_train_data[sampled_indices], y_train_data[sampled_indices]