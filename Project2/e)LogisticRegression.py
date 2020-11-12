# See medium: https://medium.com/ai-in-plain-english/comparison-between-logistic-regression-and-neural-networks-in-classifying-digits-dc5e85cd93c3

# (Read more!) Logistic regression is the same as neural network with sigmoid but without hidden layers

from nnreg.model import Model
from nnreg.trainer import Trainer
from nnreg.dataloader import DataLoader
from RegLib.HelperFunctions import get_best_dict, plot_values_with_info,plot_values_with_two_y_axis
from nnreg.config import Config
from PROJECT_SETUP import ROJECT_ROOT_DIR
from RegLib.load_save_data import load_best_checkpoint, write_json

import numpy as np
from sklearn.model_selection import ParameterGrid
# For testing:
from sklearn.linear_model  import LogisticRegression

def train(cfg, data: DataLoader, output_dir):
    cfg.dump(output_dir.joinpath("logistic_reg_mnist.yaml"))

    return Trainer().train_and_test(cfg = cfg, data_loader = data, checkpoints_path = output_dir)

# Compare to sklearn, is there a better function to compare to?:
def test(cfg, data: DataLoader, best_data_dict):
    logreg = LogisticRegression(penalty = 'l2', 
                            #C = cfg.OPTIM.L2_REG_LAMBDA,
                            fit_intercept = False,
                            solver = "sag",
                            max_iter=cfg.OPTIM.NUM_EPOCHS,
                            verbose = 0).fit(data.X_train, data.y_train)

    print("sklearn test accuracy: % .4f" % (logreg.score(data.X_test, data.y_test)))
    print("Ours test accuracy: % .4f" % (best_data_dict["Test_eval"])) 

def plot(best_data_dict):
    values_to_plot = {
        "Train_accuracy": list(best_data_dict["Train_eval"].values()),
        "Val_accuracy": list(best_data_dict["Val_eval"].values())
    }
    y2 = { "Learning_rate": list(best_data_dict["Learning_rate"].values())}

    steps = list(map(int, best_data_dict["Train_eval"].keys()))
    plot_values_with_two_y_axis(steps, values_to_plot, y2, y1_label = "Accuracy", title = "Logistic Regression on MNIST", save_fig = False)



# Make sure that the configurations are fit for logisitc regression with MNIST

activation_func = ["softmax"]
cfg = Config(config_override = [
    'OPTIM.BATCH_SIZE', 32,
    "MODEL.ACTIVATION_FUNCTIONS", activation_func,
    "MODEL.COST_FUNCTION", "ce",
    "DATA.NAME", "mnist",
    "MODEL.HIDDEN_LAYERS", [], # No hidden layers as it is regression
    "MODEL.EVAL_FUNC", "acc", # Compute accuracy
    "OUTPUT_DIR", "e)'Lo"
    ])

output_dir = ROJECT_ROOT_DIR.joinpath(cfg.OUTPUT_DIR)

data_loader = DataLoader(cfg)
train(cfg, data_loader, output_dir)
best_data_dict = get_best_dict(output_dir)

data_loader = DataLoader(cfg, one_hot_encode=False)
test(cfg, data_loader, best_data_dict)

#plot(best_data_dict)

param_grid = {
    #'OPTIM.LR_DECAY': [0.0, 0.6, 0.9],
    'OPTIM.REGULARISATION': ["l2"],
    'OPTIM.ALPHA': [0.1, 0.5, 0.9],
}