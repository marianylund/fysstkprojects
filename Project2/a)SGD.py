from PROJECT_SETUP import ROJECT_ROOT_DIR
from nnreg.config import Config
from nnreg.dataloader import DataLoader
from nnreg.trainer import Trainer
from nnreg.analysis_fun import param_search

from RegLib.load_save_data import write_json, get_previous_checkpoint_as_dict
from RegLib.HelperFunctions import get_best_dict, plot_values_with_info, plot_values_with_two_y_axis

import numpy as np
# For testing:
from sklearn.linear_model import SGDRegressor

def train(cfg, data: DataLoader, output_dir):
    cfg.dump(output_dir.joinpath("sgd.yaml"))
    return Trainer().train_and_test(cfg = cfg, data_loader = data, checkpoints_path = output_dir)

# Compare to sklearn, is there a better function to compare to?
def test(cfg, data: DataLoader, best_data_dict):
    regr = SGDRegressor(loss = 'squared_loss',
                        #penalty = cfg.OPTIM.REGULARISATION,
                        #alpha = cfg.OPTIM.ALPHA,
                        #fit_intercept = False,
                       # max_iter = cfg.OPTIM.NUM_EPOCHS * cfg.OPTIM.BATCH_SIZE,
                        #tol = None if (cfg.OPTIM.EARLY_STOP_LR_STEP == -1) else cfg.OPTIM.EARLY_STOP_LR_STEP,
                        #shuffle = cfg.SHUFFLE,
                        average = False).fit(data.X_train, data.y_train)
    regr_test_pred = regr.predict(data.X_test)

    print("sklearn MLPRegressor: R2 : % .4f, MSE : % .4f" % (-Trainer.R2(data.X_test, data.y_test), Trainer.MSE(data.y_test.ravel(), regr_test_pred))) 
    print("Ours: R2 : % .4f, MSE : % .4f" % (best_data_dict["Test_r2"], best_data_dict["Test_eval"])) 

def plot(best_data_dict):
    values_to_plot = {
        "Train_mse": list(best_data_dict["Train_eval"].values()),
        "Val_mse": list(best_data_dict["Val_eval"].values()),
        #"Train_r2": list(best_data_dict["Train_r2"].values()),
    }
    y2 = { "Learning_rate": list(best_data_dict["Learning_rate"].values())}
    steps = list(map(int, best_data_dict["Train_eval"].keys()))
    plot_values_with_two_y_axis(steps, values_to_plot, y2, y1_label = "Error", title = "SGD", save_fig = False)


config_override = [
    "MODEL.ACTIVATION_FUNCTIONS", ["identity"], # to make it linear reg
    "MODEL.HIDDEN_LAYERS", [], # No layers
    "MODEL.EVAL_FUNC", "mse",
    "MODEL.COST_FUNCTION", "mse",
    "DATA.NAME", "franke",
    'DATA.FRANKIE.N', 1000,
    'DATA.FRANKIE.NOISE', 0.1,
    "OPTIM.REGULARISATION", "none",
    "OUTPUT_DIR", "SGD_Ridge",
]

cfg = Config(config_override = config_override)
    
output_dir = ROJECT_ROOT_DIR.joinpath(cfg.OUTPUT_DIR)

# data_loader = DataLoader(cfg)
# train(cfg, data_loader, output_dir)
# best_data_dict = get_best_dict(output_dir)
# test(cfg, data_loader, best_data_dict)
# plot(best_data_dict)

# prev_check = get_previous_checkpoint_as_dict(output_dir)
# plot(prev_check)


param_grid = {
    'OPTIM.LR': [1e-3, 1e-4], 
    'OPTIM.BATCH_SIZE': [60], # try with 60 ?
    'OPTIM.LR_DECAY': [0.0, 0.6, 0.9],
    'OPTIM.ALPHA': [0.3, 0.5, 0.9, 1.0],
    'DATA.FRANKIE.P': [5],
    'OPTIM.REGULARISATION': ["l1"],
    }

#param_search(config_override, output_dir, param_grid, train, test)
