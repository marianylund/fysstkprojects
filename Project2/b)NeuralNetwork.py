from nnreg.model import Model
from nnreg.trainer import Trainer
from nnreg.dataloader import DataLoader
from nnreg.SGD import SGD
from RegLib.HelperFunctions import create_frankie_data, create_X, plot_values_with_info,plot_values_with_two_y_axis
from nnreg.config import Config
from PROJECT_SETUP import ROJECT_ROOT_DIR
from RegLib.load_save_data import load_best_checkpoint, write_json

import numpy as np
from sklearn.model_selection import ParameterGrid
# For testing:
from sklearn.neural_network import MLPRegressor

cfg = Config()
output_dir = ROJECT_ROOT_DIR.joinpath(cfg.OUTPUT_DIR)

def train(cfg, data: DataLoader, output_dir):
    cfg.dump(output_dir.joinpath("multilayer_model.yaml"))
    #cfg = Config(config_file = path_of_cfg, config_override = ["OPTIM.NUM_EPOCHS", 500])

    return Trainer().train_and_test(cfg = cfg, data_loader = data, checkpoints_path = output_dir)

# Compare to sklearn, is there a better function to compare to?:
def test(cfg, data: DataLoader, best_data_dict):
    regr = MLPRegressor(hidden_layer_sizes = cfg.MODEL.HIDDEN_LAYERS,
                        activation = "identity",
                        solver = "sgd",
                        alpha = cfg.OPTIM.L2_REG_LAMBDA,
                        batch_size = cfg.OPTIM.BATCH_SIZE,
                        learning_rate_init = cfg.OPTIM.LR,
                        max_iter = cfg.OPTIM.NUM_EPOCHS,
                        shuffle = cfg.SHUFFLE,
                        momentum = cfg.OPTIM.MOMENTUM,
                        early_stopping=True).fit(data.X_train, data.y_train.ravel())

    regr_test_pred = regr.predict(data.X_test)

    print("sklearn: R2 : % .4f, MSE : % .4f" % (SGD.R2(data.y_test.ravel(), regr_test_pred), SGD.MSE(data.y_test.ravel(), regr_test_pred))) 
    print("Ours: R2 : % .4f, MSE : % .4f" % (best_data_dict["Test_r2"], best_data_dict["Test_eval"])) 

def plot(best_data_dict):
    values_to_plot = {
        "Train_mse": list(best_data_dict["Train_eval"].values())
        #"Train_r2": list(best_data_dict["Train_r2"].values()),
    }
    y2 = { "Learning_rate": list(best_data_dict["Learning_rate"].values())}

    steps = list(best_data_dict["Train_eval"].keys())
    plot_values_with_two_y_axis(steps, values_to_plot, y2, y1_label = "Error", title = "nnreg", save_fig = False)

data_loader = DataLoader(cfg)
nn = train(cfg, data_loader, output_dir)
best_data_dict = load_best_checkpoint(output_dir)
test(cfg, data_loader, best_data_dict)
plot(best_data_dict)
