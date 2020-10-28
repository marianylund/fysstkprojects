from NeuralNetwork.MultiLayerModel import MultiLayerModel
from NeuralNetwork.trainer import Trainer
from RegLib.HelperFunctions import create_frankie_data, create_X, plot_values_with_info,plot_values_with_two_y_axis
from NeuralNetwork.config import Config
from PROJECT_SETUP import ROJECT_ROOT_DIR
from RegLib.load_save_data import load_best_checkpoint, write_json

import numpy as np
from sklearn.model_selection import ParameterGrid
# For testing:
# from sklearn.neural_network import MLPRegressor

cfg = Config()
output_dir = ROJECT_ROOT_DIR.joinpath(cfg.OUTPUT_DIR)

def train(cfg, output_dir):
    cfg.dump(output_dir.joinpath("multilayer_model.yaml"))
    #cfg = Config(config_file = path_of_cfg, config_override = ["OPTIM.NUM_EPOCHS", 500])

    x, y, z = create_frankie_data(cfg.SEED, cfg.DATA.FRANKIE.N, cfg.DATA.FRANKIE.NOISE)
    X = create_X(x, y, cfg.DATA.FRANKIE.P)
    return Trainer().train_and_test(cfg = cfg, X = X, y=z, checkpoints_path = output_dir)

nn = train(cfg, output_dir)