from PROJECT_SETUP import ROJECT_ROOT_DIR
from nnreg.config import Config
from nnreg.dataloader import DataLoader
from nnreg.trainer import Trainer

from RegLib.HelperFunctions import get_best_dict, plot_values_with_info, plot_values_with_two_y_axis

import numpy as np
from sklearn.model_selection import ParameterGrid
# For testing:
from sklearn.neural_network import MLPRegressor

cfg = Config(config_override = [
    "MODEL.ACTIVATION_FUNCTIONS", ["identity"], # to make it linear reg
    "MODEL.HIDDEN_LAYERS", [], # No layers
    "MODEL.EVAL_FUNC", "mse",
    "MODEL.COST_FUNCTION", "mse",
    "DATA.NAME", "franke",
    "OUTPUT_DIR", "SDG"
    ])
output_dir = ROJECT_ROOT_DIR.joinpath(cfg.OUTPUT_DIR)

def train(cfg, data: DataLoader, output_dir):
    cfg.dump(output_dir.joinpath("sgd.yaml"))
    return Trainer().train_and_test(cfg = cfg, data_loader = data, checkpoints_path = output_dir)

# Compare to sklearn, is there a better function to compare to?
def test(cfg, data: DataLoader, best_data_dict):
    regr = MLPRegressor(hidden_layer_sizes = (1, ), 
                        activation = 'identity',
                        solver = "sgd",
                        alpha = cfg.OPTIM.L2_REG_LAMBDA,
                        batch_size = cfg.OPTIM.BATCH_SIZE,
                        learning_rate_init = cfg.OPTIM.LR,
                        max_iter = cfg.OPTIM.NUM_EPOCHS,
                        shuffle = cfg.SHUFFLE,
                        momentum = cfg.OPTIM.MOMENTUM,
                        early_stopping=True).fit(data.X_train, data.y_train)

    regr_test_pred = regr.predict(data.X_test)

    print("sklearn: R2 : % .4f, MSE : % .4f" % (Trainer.R2(data.X_test, data.y_test), Trainer.MSE(data.y_test.ravel(), regr_test_pred))) 
    print("Ours: R2 : % .4f, MSE : % .4f" % (best_data_dict["Test_r2"], best_data_dict["Test_eval"])) 

def plot(best_data_dict):
    values_to_plot = {
        "Train_mse": list(best_data_dict["Train_eval"].values()),
        "Val_mse": list(best_data_dict["Val_eval"].values()),
        #"Train_r2": list(best_data_dict["Train_r2"].values()),
    }
    y2 = { "Learning_rate": list(best_data_dict["Learning_rate"].values())}

    steps = list(best_data_dict["Train_eval"].keys())
    plot_values_with_two_y_axis(steps, values_to_plot, y2, y1_label = "Error", title = "SGD", save_fig = False)


data_loader = DataLoader(cfg)
#train(cfg, data_loader, output_dir)
best_data_dict = get_best_dict(output_dir)
test(cfg, data_loader, best_data_dict)
plot(best_data_dict)



#param_search(output_dir)

# def param_search(output_dir):

#     param_grid = ParameterGrid({
#         'OPTIM.LR': [1e-1, 1e-2, 1e-3, 1e-4], 
#         'OPTIM.EARLY_STOP_LR_STEP': [-1.0, 1e-5, 1e-8, 1e-10]})

#     results = np.zeros(len(param_grid))
#     times = np.zeros(len(param_grid))

#     for i in range(len(param_grid)):
#         val = param_grid[i]
#         param = list(sum(val.items(), ())) # magic that flattens list of tuples
#         new_cfg = Config(config_override = param)
#         name = "".join([str(i) for i in val.values()]).replace(".", "")
#         print("Checking: ", param, " name: ", name)
#         new_output_dir = output_dir.joinpath(name)
#         sgd = train(new_cfg, new_output_dir)
#         results[i] = sgd.best_test_mse
#         times[i] = sgd.process_time
#         print("MSE: ", results[i], ", time: ", times[i])
    
#     best_mse_i = np.argmin(results)
#     results_dict = {
#         "best_index":  best_mse_i,
#         "best_mse": results[best_mse_i],
#         "best_param": param_grid[best_mse_i],
#         "best_time": times[best_mse_i],
#         "param" : param_grid,
#         "results": results,
#         "times": times
#     }
#     write_json(results_dict, output_dir.joinpath("param_search_results.json"))
#     print("Best mse: ", results[best_mse_i], " with param: ", param_grid[best_mse_i], ", time: ", times[best_mse_i])
    

