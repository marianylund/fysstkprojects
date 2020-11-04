# See medium: https://medium.com/ai-in-plain-english/comparison-between-logistic-regression-and-neural-networks-in-classifying-digits-dc5e85cd93c3

# (Read more!) Logistic regression is the same as neural network with sigmoid but without hidden layers

from nnreg.model import Model
from nnreg.trainer import Trainer
from nnreg.dataloader import DataLoader
from nnreg.SGD import SGD
from RegLib.HelperFunctions import plot_values_with_info,plot_values_with_two_y_axis
from nnreg.config import Config
from PROJECT_SETUP import ROJECT_ROOT_DIR
from RegLib.load_save_data import load_best_checkpoint, write_json

import numpy as np
from sklearn.model_selection import ParameterGrid
# For testing:
from sklearn.linear_model  import LogisticRegression


# Make sure that the configurations are fit for classification with MNIST
act_with_softmax = Config().MODEL.ACTIVATION_FUNCTIONS
act_with_softmax[-1] = "sigmoid"
cfg = Config(config_override = [
    "MODEL.ACTIVATION_FUNCTIONS", act_with_softmax,
    "MODEL.COST_FUNCTION", "ce",
    "DATA.NAME", "mnist",
    #"OUTPUT_DIR", "mnist_classification"
    ])
output_dir = ROJECT_ROOT_DIR.joinpath(cfg.OUTPUT_DIR)

def train(cfg, data: DataLoader, output_dir):
    #cfg = Config(config_file = path_of_cfg, config_override = ["OPTIM.NUM_EPOCHS", 500])
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

    clf = MLPClassifier(verbose = False, validation_fraction=0.2).fit(data.X_train, data.y_train)

    #clf_test_pred = clf.predict(data.X_test)
    print("Sklearn activation: ", clf.out_activation_)
    print("X_shape",  data.X_test.shape)
    new_data = data.X_test[0].reshape(-1, 1).T
    print("new_data", new_data.shape, data.y_test[0])
    print("Sklearn activation: ", clf.predict(new_data))
    print("sklearn accuracy: % .4f" % (clf.score(data.X_test, data.y_test)))
    print("Ours accuracy: % .4f" % (best_data_dict["Test_eval"])) 

def plot(best_data_dict):
    values_to_plot = {
        "Train_accuracy": list(best_data_dict["Train_eval"].values())
    }
    y2 = { "Learning_rate": list(best_data_dict["Learning_rate"].values())}

    steps = [x for x in range(best_data_dict["Step"] + 1)]
    plot_values_with_two_y_axis(steps, values_to_plot, y2, title = "Classification on MNIST", save_fig = False)

data_loader = DataLoader(cfg)
nn = train(cfg, data_loader, output_dir)
best_data_dict = load_best_checkpoint(output_dir)
#test(cfg, data_loader, best_data_dict)
plot(best_data_dict)