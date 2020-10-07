import numpy as np

class RegressionResults():
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
            # Get variance between points of diff models, not betweeen points themselves
            # They are supposed to be the same
            self.variance_beta = 0
            self.bias = self.get_bias(y_data, self.beta_optimal)
            assert self.mse >= (self.bias + self.variance_beta), ('[{}] {} >= {} + {} = {}'.format(index, self.mse, self.bias, self.variance_beta, self.bias+self.variance_beta))
        else:
            self.r2_avg[index] = self.R2(y_data, self.beta_optimal)
            self.mse_avg[index] = self.MSE(y_data, self.beta_optimal)
            self.variance_beta_avg[index] = np.var(self.beta_optimal)
            self.bias_avg[index] = self.get_bias(y_data, self.beta_optimal)
            assert self.mse_avg[index] >= (self.bias_avg[index] + self.variance_beta_avg[index]), ('[{}] {} >= {} + {} = {}'.format(index, self.mse_avg[index], self.bias_avg[index], self.variance_beta_avg[index], self.bias_avg[index]+self.variance_beta_avg[index]))


    def get(self):
        return self.r2, self.mse, self.variance_beta, self.bias
    
    def get_avg(self):
        return self.r2_avg, self.mse_avg, self.variance_beta_avg, self.bias_avg
    
    @staticmethod
    def R2(y_data, y_model):
        return 1 - np.sum((y_data - y_model)**2) / np.sum((y_data - np.mean(y_data))**2)

    @staticmethod
    def MSE(y_data, y_model):
        return np.mean((y_data - y_model)**2)

    @staticmethod
    def get_bias(y_data, y_model):
        return np.mean((y_data - np.mean(y_model, axis=1, keepdims=True))**2)
