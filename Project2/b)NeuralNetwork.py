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

def train(cfg, output_dir, just_split = False):
    cfg.dump(output_dir.joinpath("multilayer_model.yaml"))
    #cfg = Config(config_file = path_of_cfg, config_override = ["OPTIM.NUM_EPOCHS", 500])

    x, y, z = create_frankie_data(cfg.SEED, cfg.DATA.FRANKIE.N, cfg.DATA.FRANKIE.NOISE)
    X = create_X(x, y, cfg.DATA.FRANKIE.P)
    if just_split:
        return Trainer().split_and_scale_train_test(X = X, y=z, test_size = cfg.TEST_SIZE)
    else:
        return Trainer().train_and_test(cfg = cfg, X = X, y=z, checkpoints_path = output_dir)

# Compare to sklearn, is there a better function to compare to?:
def test(cfg, other_model, best_data_dict):
    regr = MLPRegressor(hidden_layer_sizes = cfg.MODEL.SHAPE,
                        activation = "identity",
                        solver = "sgd",
                        alpha = cfg.OPTIM.L2_REG_LAMBDA,
                        batch_size = cfg.OPTIM.BATCH_SIZE,
                        learning_rate_init = cfg.OPTIM.LR,
                        max_iter = cfg.OPTIM.NUM_EPOCHS,
                        shuffle = cfg.SHUFFLE,
                        momentum = cfg.OPTIM.MOMENTUM,
                        early_stopping=True).fit(other_model.X_train, other_model.y_train.ravel())

    regr_test_pred = regr.predict(other_model.X_test)

    print("sklearn: R2 : % .4f, MSE : % .4f" % (SGD.R2(other_model.X_test, other_model.y_test), SGD.MSE(other_model.y_test.ravel(), regr_test_pred))) 
    print("Ours: R2 : % .4f, MSE : % .4f" % (best_data_dict["Test_r2"], best_data_dict["Test_mse"])) 

def plot(best_data_dict):
    values_to_plot = {
        "Train_mse": list(best_data_dict["Train_mse"].values())
        #"Train_r2": list(best_data_dict["Train_r2"].values()),
    }
    y2 = { "Learning_rate": list(best_data_dict["Learning_rate"].values())}

    steps = [x for x in range(best_data_dict["Step"] + 1)]
    plot_values_with_two_y_axis(steps, values_to_plot, y2, title = "NeuralNetwork", save_fig = False)

nn = train(cfg, output_dir, just_split = True)
best_data_dict = load_best_checkpoint(output_dir)
test(cfg, nn, best_data_dict)
plot(best_data_dict)
