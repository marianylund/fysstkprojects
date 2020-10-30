# Useful info: https://towardsdatascience.com/implementing-different-activation-functions-and-weight-initialization-methods-using-python-c78643b9f20f

from NeuralNetwork.MultiLayerModel import MultiLayerModel
from NeuralNetwork.trainer import Trainer
from NeuralNetwork.SGD import SGD
from RegLib.HelperFunctions import create_frankie_data, create_X, plot_values_with_info,plot_values_with_two_y_axis
from NeuralNetwork.config import Config
from PROJECT_SETUP import ROJECT_ROOT_DIR
from RegLib.load_save_data import load_best_checkpoint, write_json

import numpy as np
from sklearn.model_selection import ParameterGrid
# For testing:
from sklearn.neural_network import MLPRegressor

cfg = Config()
output_dir = ROJECT_ROOT_DIR.joinpath(cfg.OUTPUT_DIR)