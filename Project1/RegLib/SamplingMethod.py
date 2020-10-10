import numpy as np
from sklearn.model_selection import train_test_split
from RegLib.RegressionMethod import RegressionMethod, RegressionType

class SamplingMethod:
    
    def train_and_test(self, X, y, perm_index = [-1], model_type = RegressionType.OLS, alpha = 0.0, test_size  = 0.2, shuffle = False, normalize=True):
        self.split_and_scale_train_test(X, y, perm_index, test_size, shuffle, normalize)
        self.model = RegressionMethod().fit(self.X_train, self.y_train, model_type, alpha)
        self.test_model(self.model, self.X_test, self.y_test)
        return self

    def split_and_scale_train_test(self, X, y, perm_index = [-1], test_size  = 0.2, shuffle = False, normalize=True):
        assert X.shape[0] == y.shape[0], ("X.shape[0] and y.shape[0] needs to be the same length, but: " + str(X.shape[0]) + " != " + str(y.shape[0]))
        if(len(perm_index) > 1):
            X = X[perm_index]
            y = y[perm_index]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, shuffle=shuffle)
        if normalize:
            self.X_train, self.X_test = SamplingMethod.scale_standard(self.X_train, self.X_test)
        # Force the correct shape:
        self.y_test.shape = (self.y_test.shape[0], 1)
        self.y_train.shape = (self.y_train.shape[0], 1)
        return self

    def test_model(self, model, X_data, y_data):
        y_pred = model.get_y_pred(X_data)
        self.r2 = self.R2(y_data, y_pred)
        self.mse = self.MSE(y_data, y_pred)
        self.bias = self.get_bias(y_data, y_pred)
        return self
    
    def __repr__(self):
        return ', '.join("%s: %s" % item for item in vars(self).items())
    
    @staticmethod
    def R2(y_data, y_pred):
        return 1 - np.sum((y_data - y_pred)**2) / np.sum((y_data - np.mean(y_data))**2)

    @staticmethod
    def MSE(y_data, y_pred):
        return np.mean((y_data - y_pred)**2)

    @staticmethod
    def get_bias(y_data, y_pred): # TODO do you need need to take mean?
        if(len(y_pred.shape) == 1):
            return np.mean((y_data - y_pred)**2)
        return np.mean((y_data - np.mean(y_pred, axis=1, keepdims=True))**2)

    @staticmethod
    def scale_standard(train_data, test_data):
        data_mean = np.mean(train_data[:,1:], axis = 0)
        data_std = np.std(train_data[:,1:], axis = 0)
        train_data_scaled = train_data
        test_data_scaled = test_data
        train_data_scaled[:,1:] = np.divide((train_data[:,1:] - data_mean), data_std)
        test_data_scaled[:,1:] = np.divide((test_data[:,1:] - data_mean), data_std)
        
        return train_data_scaled, test_data_scaled
    
