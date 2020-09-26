import numpy as np
from sklearn.model_selection import train_test_split

class SamplingData:
    def __init__(self, X, y, model):
        self.X = X
        self.y = y
        self.model = model
    
    def fit(self):
        self.split_and_scale_train_test()
        self.model.find_beta()
        self.model.find_optimal_beta()
        self.model.test_model()

    def split_and_scale_train_test(self, test_size  = 0.2, shuffle = False, normalize=True):
        assert self.X.shape[0] == self.y.shape[0], ("X.shape[0] and y.shape[0] needs to be the same length, but: " + str(self.X.shape[0]) + " != " + str(self.y.shape[0]))

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, shuffle=shuffle)
        if normalize:
            X_train, X_test = SamplingData.scale_standard(X_train, X_test)

        self.model.set_data(X_train, X_test, y_train, y_test)
    
    def __repr__(self):
        return ', '.join("%s: %s" % item for item in vars(self).items())

    @staticmethod
    def scale_standard(train_data, test_data):
        data_mean = np.mean(train_data[:,1:], axis = 0)
        data_std = np.std(train_data[:,1:], axis = 0)
        train_data_scaled = train_data
        test_data_scaled = test_data
        train_data_scaled[:,1:] = np.divide((train_data[:,1:] - data_mean), data_std)
        test_data_scaled[:,1:] = np.divide((test_data[:,1:] - data_mean), data_std)
        
        return train_data_scaled, test_data_scaled
    
