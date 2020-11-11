from PROJECT_SETUP import ROJECT_ROOT_DIR
from nnreg.config import Config
from nnreg.dataloader import DataLoader
from nnreg.trainer import Trainer
from time import time

from RegLib.load_save_data import write_json
from RegLib.HelperFunctions import get_best_dict, plot_values_with_info, plot_values_with_two_y_axis

import numpy as np
from sklearn.model_selection import ParameterGrid
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

    steps = list(best_data_dict["Train_eval"].keys())
    plot_values_with_two_y_axis(steps, values_to_plot, y2, y1_label = "Error", title = "SGD", save_fig = False)

def param_search(configs, output_dir):

    start_time = time()

    param_grid = ParameterGrid({
        'OPTIM.LR': [1e-2, 1e-3, 1e-4], 
        'DATA.FRANKIE.P': [5, 10, 15],
        'OPTIM.BATCH_SIZE': [20, 60, 120],
        })

    total_combinations = len(param_grid)
    results = np.zeros(total_combinations)
    times = np.zeros((total_combinations, 2)) # min, sec
    print("Total combinations: ", total_combinations)

    for i in range(total_combinations):
        val = param_grid[i]
        param = list(sum(val.items(), ())) # magic that flattens list of tuples

        name = "".join([str(i) for i in val.values()]).replace(".", "")
        print("Checking: ", param, " name: ", name)
        new_output_dir = output_dir.joinpath(name)

        ind_of_output = configs.index("OUTPUT_DIR")
        configs[ind_of_output + 1] = configs[ind_of_output] + "\\" + name

        new_cfg = Config(config_override = configs + param)

        data_loader = DataLoader(new_cfg)
        train(new_cfg, data_loader, new_output_dir)
        best_data_dict = get_best_dict(new_output_dir)
        results[i] = best_data_dict["Test_eval"]
        times[i] = best_data_dict["Proccess_time"]
        test(new_cfg, data_loader, best_data_dict)

        print("Time passed: " + str(divmod(time() - start_time, 60)))
        
    
    best_eval_i = np.argmin(results)
    results_dict = {
        "best_index":  best_eval_i,
        "best_eval": results[best_eval_i],
        "best_param": param_grid[best_eval_i],
        "best_time": times[best_eval_i],
        "param" : param_grid,
        "results": results,
        "times": times
    }
    write_json(results_dict, output_dir.joinpath("param_search_results.json"))
    print("Best eval: ", results[best_eval_i], " with param: ", param_grid[best_eval_i], ", time: ", times[best_eval_i])
    

config_override = [
    "MODEL.ACTIVATION_FUNCTIONS", ["identity"], # to make it linear reg
    "MODEL.HIDDEN_LAYERS", [], # No layers
    "MODEL.EVAL_FUNC", "mse",
    "MODEL.COST_FUNCTION", "mse",
    "DATA.NAME", "franke",
    "OPTIM.REGULARISATION", "none",
    "OUTPUT_DIR", "SDG"
]

cfg = Config(config_override = config_override)
    
output_dir = ROJECT_ROOT_DIR.joinpath(cfg.OUTPUT_DIR)

data_loader = DataLoader(cfg)
train(cfg, data_loader, output_dir)
best_data_dict = get_best_dict(output_dir)
test(cfg, data_loader, best_data_dict)
plot(best_data_dict)

#param_search(config_override, output_dir)
