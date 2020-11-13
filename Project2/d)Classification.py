from PROJECT_SETUP import ROJECT_ROOT_DIR
from nnreg.model import Model
from nnreg.trainer import Trainer
from nnreg.dataloader import DataLoader
from nnreg.config import Config
from nnreg.analysis_fun import show_heatmap, get_min_value, unpack, get_paths_of_results_where, plot_values_with_steps_and_info, param_search, train_save_configs, plot_lr_tran_val

from RegLib.HelperFunctions import create_frankie_data, create_X, plot_values_with_info,plot_values_with_two_y_axis
from RegLib.load_save_data import get_previous_checkpoint_as_dict, load_best_checkpoint, write_json, get_previous_checkpoints, load_data_as_dict

# For testing:
from sklearn.neural_network import MLPClassifier

# For Analysis:
from math import inf, isnan
import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Compare to sklearn, is there a better function to compare to?:
def test(cfg, data: DataLoader, best_data_dict):
    clf = MLPClassifier(hidden_layer_sizes = cfg.MODEL.HIDDEN_LAYERS, verbose = False, validation_fraction=0.2, early_stopping=True).fit(data.X_train, data.y_train)

    print("sklearn accuracy: % .4f" % (clf.score(data.X_test, data.y_test)))
    print("Ours accuracy: % .4f" % (best_data_dict["Test_eval"])) 

# Make sure that the configurations are fit for classification with MNIST
config_override = [
    'OPTIM.BATCH_SIZE', 32,
    'OPTIM.REGULARISATION', "none",
    'OPTIM.NUM_EPOCHS', '30',
    'OPTIM.LR', 1e-2, 
    'OPTIM.EARLY_STOP_LR_STEP', 6000.0, 
    "MODEL.HIDDEN_LAYERS", [100, 20],
    "MODEL.ACTIVATION_FUNCTIONS", ["leaky_relu", "leaky_relu", "softmax"],
    'MODEL.WEIGHT_INIT', 'xavier',
    'MODEL.LEAKY_SLOPE',  0.1,
    "MODEL.EVAL_FUNC", "acc",
    "MODEL.COST_FUNCTION", "ce",
    "DATA.NAME", "mnist",
    "DATA.MNIST.BINARY", [], # all classes
    "OUTPUT_DIR", "d)MNIST_Num_Of_Nodes"
]

cfg = Config(config_override = config_override)
output_dir = ROJECT_ROOT_DIR.joinpath(cfg.OUTPUT_DIR)

data_loader = DataLoader(cfg)
train_save_configs(cfg, data_loader, output_dir)
best_data_dict = load_best_checkpoint(output_dir)
test(cfg, data_loader, best_data_dict)
plot_lr_tran_val(best_data_dict)










# ------------------------Parameter search-----------------------------------

param_grid = {
    #'OPTIM.LR_DECAY': [0.0, 0.6, 0.9],
    'OPTIM.REGULARISATION': ["l2"],
    'OPTIM.ALPHA': [0.1, 0.5, 0.9],
}

# change OUTPUT DIR 
param_grid = {
    "MODEL.HIDDEN_LAYERS": [[200, 100, 20], [10, 200, 10]],
    'MODEL.ACTIVATION_FUNCTIONS': [["tanh", "tanh", "leaky_relu", "softmax"], ["leaky_relu", "leaky_relu", "leaky_relu", "softmax"]],
}

#param_search(config_override, output_dir, param_grid, train, test)

# ------------------------Analysis of results-----------------------------------
# Regularisation

def plot_l2():
    path_to_results = Path("Results", "d)MNISTClass_Regularisation")
    all_dir = [x for x in path_to_results.iterdir() if x.is_dir()]
    values_to_plot = {}
    steps_to_plot = {}
    for i in range(len(all_dir)):
        d = all_dir[i]
        cfg = Config(config_file = Path(d).joinpath("classification_mnist_model.yaml"))
        
        last = get_previous_checkpoint_as_dict(d)

        new_key = f'Alpha: {cfg.OPTIM.ALPHA}'
        values_to_plot[new_key] = list(last["Val_eval"].values())
        steps_to_plot[new_key] = list(map(int, last["Val_eval"].keys()))

    info_to_add = {}
    ylimit = None #(0.01, 0.04) #
    xlimit = None #(0, 50000) #
    save_fig = True
    plot_values_with_steps_and_info(steps_to_plot, values_to_plot, title = "L2 Regularisation on MNIST", xlimit = xlimit, ylabel = "Accuracy",  info_to_add = info_to_add, ylimit = ylimit, save_fig = save_fig)

#plot_l2()


def analyse_weight_init_activ(leaky = False):
    path_to_results = Path("Results", "d)MNISTClass_Weigth_Act")
    all_dir = [x for x in path_to_results.iterdir() if x.is_dir()]
    values_to_plot = {}
    steps_to_plot = {}
    for i in range(len(all_dir)):
        d = all_dir[i]
        cfg = Config(config_file = Path(d).joinpath("classification_mnist_model.yaml"))
        if (leaky and cfg.MODEL.ACTIVATION_FUNCTIONS[0] == "leaky_relu") or (not leaky and cfg.MODEL.ACTIVATION_FUNCTIONS[0] != "leaky_relu"):
            last = get_previous_checkpoint_as_dict(d)

            weight_init = cfg.MODEL.WEIGHT_INIT
            act = cfg.MODEL.ACTIVATION_FUNCTIONS[0]

            new_key = f"{weight_init}_{act}"
            values_to_plot[new_key] = list(last["Val_eval"].values())
            steps_to_plot[new_key] = list(map(int, last["Val_eval"].keys()))

    info_to_add = {}
    ylimit = None #(0.01, 0.04) #
    xlimit = None #(0, 50000) #
    save_fig = True
    plot_values_with_steps_and_info(steps_to_plot, values_to_plot, title = "MNIST Weight Init and Activations", xlimit = xlimit, ylabel = "Accuracy",  info_to_add = info_to_add, ylimit = ylimit, save_fig = save_fig)

#analyse_weight_init_activ(leaky = False)

def analyse_nodes_func():
    path_to_results = Path("Results", "d)MNIST_Num_Of_Nodes")
    all_dir = [x for x in path_to_results.iterdir() if x.is_dir()]
    values_to_plot = {}
    steps_to_plot = {}
    for i in range(len(all_dir)):
        d = all_dir[i]
        cfg = Config(config_file = Path(d).joinpath("classification_mnist_model.yaml"))
        last = get_previous_checkpoint_as_dict(d)
        hidden_layers = cfg.MODEL.HIDDEN_LAYERS
        act = cfg.MODEL.ACTIVATION_FUNCTIONS[:-1]

        new_key = f"{hidden_layers}_{act}"
        values_to_plot[new_key] = list(last["Val_eval"].values())
        steps_to_plot[new_key] = list(map(int, last["Val_eval"].keys()))

    info_to_add = {}
    ylimit = None #(0.01, 0.04) #
    xlimit = (0, 10000) #
    save_fig = False
    plot_values_with_steps_and_info(steps_to_plot, values_to_plot, title = "MNIST Leaky ReLU", xlimit = xlimit, ylabel = "Accuracy",  info_to_add = info_to_add, ylimit = ylimit, save_fig = save_fig)

#analyse_nodes_func()