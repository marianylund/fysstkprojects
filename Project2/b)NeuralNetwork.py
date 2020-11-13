from nnreg.model import Model
from nnreg.trainer import Trainer
from nnreg.dataloader import DataLoader
from RegLib.HelperFunctions import create_frankie_data, create_X, plot_values_with_info,plot_values_with_two_y_axis
from nnreg.config import Config
from PROJECT_SETUP import ROJECT_ROOT_DIR
from RegLib.load_save_data import load_best_checkpoint, write_json
from nnreg.analysis_fun import get_paths_of_results_where, plot_values_with_steps_and_info, param_search, train_save_configs, plot_lr_tran_val

import numpy as np
from sklearn.model_selection import ParameterGrid
# For testing:
from sklearn.neural_network import MLPRegressor

# For Analysis:
from math import inf
import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Compare to sklearn, is there a better function to compare to?:
def test(cfg, data: DataLoader, best_data_dict):
    regr = MLPRegressor(hidden_layer_sizes = cfg.MODEL.HIDDEN_LAYERS,
                        activation = "identity",
                        solver = "sgd",
                        early_stopping=True).fit(data.X_train, data.y_train.ravel())

    regr_test_pred = regr.predict(data.X_test)

    print("sklearn: R2 : % .4f, MSE : % .4f" % (Trainer.R2(data.y_test.ravel(), regr_test_pred), Trainer.MSE(data.y_test.ravel(), regr_test_pred))) 
    print("Ours: R2 : % .4f, MSE : % .4f" % (best_data_dict["Test_r2"], best_data_dict["Test_eval"])) 

hidden_layers = [10, 5]

config_override = [
    "MODEL.ACTIVATION_FUNCTIONS", ["sigmoid", "sigmoid", "identity"], # will need to vary activations in c)
    "MODEL.HIDDEN_LAYERS", hidden_layers,  # this will need to vary
    "MODEL.WEIGHT_INIT", "random", # this will need to vary in c) # {'random', 'he', 'xavier', 'zeros'}
    "MODEL.EVAL_FUNC", "mse",
    "MODEL.COST_FUNCTION", "mse",
    "OPTIM.REGULARISATION", "none",
    "OPTIM.BATCH_SIZE", 60,
    "OPTIM.LR", 1e-3, # for now just concentrate on this lr
    "DATA.NAME", "franke",
    'DATA.FRANKIE.P', 5,
    'DATA.FRANKIE.N', 1000,
    'DATA.FRANKIE.NOISE', 0.1,
    "OUTPUT_DIR", "Testb)NeuralNetwork",
]

cfg = Config(config_override = config_override)
output_dir = ROJECT_ROOT_DIR.joinpath(cfg.OUTPUT_DIR)

data_loader = DataLoader(cfg)
train_save_configs(cfg, data_loader, output_dir)
best_data_dict = load_best_checkpoint(output_dir)
test(cfg, data_loader, best_data_dict)
plot_lr_tran_val(best_data_dict)

# Test with different number of hidden layers

# hidden_layers = [5]
# sklearn: R2 : 0.8148, MSE :  0.0167
# Ours: R2 :  0.8842, MSE :  0.0104
# Time: 18:37

# hidden_layers = [10, 5]
# sklearn: R2 :  0.8560, MSE :  0.0130
# Ours: R2 :  0.8893, MSE :  0.0100
# Time: 53:28

# param_grid = {
#     'MODEL.HIDDEN_LAYERS': [[5]], 
# }

#param_search(config_override, output_dir, param_grid, train, test)

