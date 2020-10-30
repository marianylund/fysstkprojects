import numpy as np
from random import random, seed
import pathlib

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import mean_squared_error

from RegLib.SamplingMethod import SamplingMethod
from RegLib.RegressionMethod import RegressionType
from RegLib.HelperFunctions import confidence_interval, create_frankie_data, create_X
from RegLib.load_save_data import *
from PROJECT_SETUP import SEED, CHECKPOINTS_DIR
from NeuralNetwork.config import Config
from NeuralNetwork.trainer import Trainer
from NeuralNetwork.MultiLayerModel import MultiLayerModel

error_tolerance = 1e-10

def create_test_data():
    N = 20
    noise_strength = 0.1
    p = 1

    x, y, z = create_frankie_data(SEED, N, noise_strength)
    X = create_X(x, y, p)
    return X, z

def test_saving_and_loading_model(X, z):
    print("Testing Saving Model")

    run_info = {'seed': SEED,
        # 'N': N,
        # 'noise_strength': noise_strength,
        # 'p': p,
        }
    
    perm_index = np.random.permutation(len(z))
    sampling = SamplingMethod().train_and_test(X, z, perm_index = perm_index, model_type = RegressionType.OLS)
    full_dict = add_more_info_to_dict(sampling.to_dict(), run_info)

    save_checkpoint(full_dict, CHECKPOINTS_DIR.joinpath("test", "test_data.json"))
    loaded_data = load_data_as_dict(CHECKPOINTS_DIR.joinpath("test", "test_data.json"))
    loaded_sampling = SamplingMethod().from_dict(loaded_data)
    assert np.equal(loaded_sampling.model.beta, sampling.model.beta).all(), "Loaded incorrect model's beta"

def test_regression_learning_scikit(X, y):
    print("Testing Loading Saved Model")

    scaler = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    regr = MLPRegressor(early_stopping=True).fit(X_train, y_train)
    print("R2 : % .4f, MSE : % .4f" % (regr.score(X_test, y_test), mean_squared_error(y_test, regr.predict(X_test)))) 

def test_classification_scikit():
    print("Testing Loading Saved Model")

    X = [[0., 0.], [1., 1.]]
    y = [0, 1]
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

    clf.fit(X, y) # array([1, 0])

def test_classification_softmax_scikit():
    print("Testing multi-classification with softmax")
    X = [[0., 0.], [1., 1.]]
    y = [[0, 1], [1, 1]]
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)

    clf.fit(X, y)
    clf.predict([[1., 2.]]) #array([[1, 1]])
    clf.predict([[0., 0.]]) #array([[0, 1]])

def test_sgd(X, y):
    return
    #train_sgd(cfg, X, y)

def test_accuracy():
    y_data = np.ndarray([[0, 0, 1, 0]])
    y_pred = np.ndarray([[1, 0, 0, 0]])
    acc = MultiLayerModel.calculate_accuracy(y_data, y_pred)
    assert acc == 0, acc

    acc = MultiLayerModel.calculate_accuracy(y_data, y_data)
    assert acc == 1, acc

    
    

if __name__ == "__main__":
    print("Start tests for project 2")
    #X, z = create_test_data()
    #test_saving_and_loading_model(X, z)
    #test_regression_learning_scikit(X, z)
    #test_sgd(X, z)
    test_accuracy()

    print("All tests have passed")
