import numpy as np
from SamplingData import SamplingData
from RegressionModel import ModelResults

class CrossValidationKFold(SamplingData):

    def __init__(self, X, y, model, kfolds = 5):
        super(CrossValidationKFold, self).__init__(X, y, model)
        self.kfolds = kfolds
    
    def fit(self):
        #self.split_and_scale_train_test()
        return self.run_k_fold_validation()
    
    def run_k_fold_validation(self):
        assert self.X.shape[0] == self.y.shape[0], ("X.shape[0] and y.shape[0] needs to be the same length, but: " + str(self.X.shape[0]) + " != " + str(self.y.shape[0]))
            
        X_fold_indices = [x for x in range(self.X.shape[0])]
        X_fold_indices = np.reshape(X_fold_indices, (self.kfolds, -1))
        k_indices = [x for x in range(self.kfolds)]

        cross_validation_train_results = ModelResults(self.kfolds)
        cross_validation_test_results = ModelResults(self.kfolds)

        for fold in range(self.kfolds):
            X_indices = X_fold_indices[np.delete(k_indices, fold)].reshape(-1)
            X_train, X_test = SamplingData.scale_standard(self.X[X_indices], self.X[X_fold_indices[fold]])
            self.model.set_data(X_train, X_test, self.y[X_indices], self.y[X_fold_indices[fold]])
            
            self.model.train_and_test()

            cross_validation_train_results.set_results(self.model.train_results, fold)
            cross_validation_test_results.set_results(self.model.test_results, fold)

        cross_validation_train_results.average_out_results()
        cross_validation_test_results.average_out_results()

        return cross_validation_train_results, cross_validation_test_results
        
