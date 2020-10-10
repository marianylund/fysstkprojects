import numpy as np
from RegLib.SamplingMethod import SamplingMethod
from RegLib.RegressionMethod import RegressionMethod, RegressionType

class CrossValidationKFold(SamplingMethod):

    def __init__(self, kfolds = 5):
        self.kfolds = kfolds
    
    def train_and_test(self, X, y, perm_index = -1, model_type = RegressionType.OLS, alpha = 0.0, test_size  = 0.2, shuffle = False, normalize=True):
        if(len(perm_index) > 1):
            X = X[perm_index]
            y = y[perm_index]
        return self.run_k_fold_validation(X, y, model_type, alpha = alpha)
    
    def run_k_fold_validation(self, X, y, model_type, alpha = 0.0):
        assert X.shape[0] == y.shape[0], ("X.shape[0] and y.shape[0] needs to be the same length, but: " + str(X.shape[0]) + " != " + str(y.shape[0]))
            
        X_fold_indices = [x for x in range(X.shape[0])]
        X_fold_indices = np.reshape(X_fold_indices, (self.kfolds, -1))
        k_indices = [x for x in range(self.kfolds)]

        y_pred = np.empty((len(X_fold_indices[0]), self.kfolds))
        y_pred_train = np.empty((len(X_fold_indices[0]) * (self.kfolds - 1), self.kfolds))

        for fold in range(self.kfolds):
            X_indices = X_fold_indices[np.delete(k_indices, fold)].reshape(-1)
            X_train, X_test = SamplingMethod.scale_standard(X[X_indices], X[X_fold_indices[fold]])
            
            y_train = y[X_indices]
            y_test = y[X_fold_indices[fold]]
            y_test.shape = (y_test.shape[0], 1)
            y_train.shape = (y_train.shape[0], 1)

            model = RegressionMethod().fit(X_train, y_train, model_type, alpha)

            y_pred[:, fold] = model.get_y_pred(X_test).ravel()
            y_pred_train[:, fold] = model.get_y_pred(X_train).ravel()

        self.r2 = self.R2(y_test, y_pred)
        self.mse = self.MSE(y_test, y_pred)
        self.bias = self.get_bias(y_test, y_pred)
        self.var = self.get_variance(y_pred)

        self.r2_train = self.R2(y_train, y_pred_train)
        self.mse_train = self.MSE(y_train, y_pred_train)
        self.bias_train = self.get_bias(y_train, y_pred_train)
        self.var_train = self.get_variance(y_pred_train)

        return self
        
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