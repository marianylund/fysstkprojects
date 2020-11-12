from nnreg.model import Model
from nnreg.trainer import Trainer
from nnreg.dataloader import DataLoader
from RegLib.HelperFunctions import create_frankie_data, create_X, plot_values_with_info,plot_values_with_two_y_axis
from nnreg.config import Config
from nnreg.analysis_fun import param_search
from PROJECT_SETUP import ROJECT_ROOT_DIR
from RegLib.load_save_data import load_best_checkpoint, write_json

from pathlib import Path
import numpy as np
import sys
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import confusion_matrix # TODO: use it
# For testing:
from sklearn.neural_network import MLPClassifier

from os import system

def train(cfg, data: DataLoader, output_dir):
    cfg.dump(output_dir.joinpath("classification_mnist_model.yaml"))
    return Trainer().train_and_test(cfg = cfg, data_loader = data, checkpoints_path = output_dir)

# Compare to sklearn, is there a better function to compare to?:
def test(cfg, data: DataLoader, best_data_dict):
    # clf = MLPClassifier(hidden_layer_sizes = cfg.MODEL.HIDDEN_LAYERS,
    #                     activation = "logistic",
    #                     solver = "sgd",
    #                     alpha = cfg.OPTIM.L2_REG_LAMBDA,
    #                     batch_size = cfg.OPTIM.BATCH_SIZE,
    #                     learning_rate_init = cfg.OPTIM.LR,
    #                     max_iter = 100,#cfg.OPTIM.NUM_EPOCHS,
    #                     shuffle = cfg.SHUFFLE,
    #                     momentum = 0.0,
    #                     early_stopping=True).fit(data.X_train, data.y_train)

    clf = MLPClassifier(hidden_layer_sizes = cfg.MODEL.HIDDEN_LAYERS, verbose = False, validation_fraction=0.2, early_stopping=True).fit(data.X_train, data.y_train)

    #clf_test_pred = clf.predict(data.X_test)
    #print("Sklearn activation: ", clf.out_activation_)
    #print("X_shape",  data.X_test.shape)
    #new_data = data.X_test[0].reshape(-1, 1).T
    #print("new_data", new_data.shape, data.y_test[0])
    #print("Sklearn activation: ", clf.predict(new_data))
    print("sklearn accuracy: % .4f" % (clf.score(data.X_test, data.y_test)))
    print("Ours accuracy: % .4f" % (best_data_dict["Test_eval"])) 

def plot(best_data_dict):
    values_to_plot = {
        "Train_accuracy": list(best_data_dict["Train_eval"].values()),
        "Val_mse": list(best_data_dict["Val_eval"].values()),
    }
    y2 = { "Learning_rate": list(best_data_dict["Learning_rate"].values())}

    steps = list(map(int, best_data_dict["Train_eval"].keys()))
    plot_values_with_two_y_axis(steps, values_to_plot, y2, title = "Classification on MNIST", save_fig = False)

# Make sure that the configurations are fit for classification with MNIST

config_override = [
    'OPTIM.BATCH_SIZE', 60,
    'OPTIM.REGULARISATION', "none",
    'OPTIM.NUM_EPOCHS', '20',
    "MODEL.HIDDEN_LAYERS", [100, 20],
    "MODEL.ACTIVATION_FUNCTIONS", ["sigmoid", "sigmoid", "softmax"],
    "MODEL.EVAL_FUNC", "acc",
    "MODEL.COST_FUNCTION", "ce",
    "DATA.NAME", "mnist",
    "DATA.MNIST.BINARY", [], # all classes
    "OUTPUT_DIR", "d)MNISTClass_LR_Batch_size"
]

cfg = Config(config_override = config_override)
output_dir = ROJECT_ROOT_DIR.joinpath(cfg.OUTPUT_DIR)


# data_loader = DataLoader(cfg)
# nn = train(cfg, data_loader, output_dir)
# best_data_dict = load_best_checkpoint(output_dir)
# test(cfg, data_loader, best_data_dict)
# plot(best_data_dict)

param_grid = {
    'OPTIM.LR': [1e-2, 1e-3, 1e-4], 
    'OPTIM.LR_DECAY': [0.0, 0.6, 0.9],
    'OPTIM.BATCH_SIZE': [32, 60],
}



#param_search(config_override, output_dir, param_grid, train, test)

#save_terminal_history(output_dir.joinpath("terminal_output"))

# Analyse results




