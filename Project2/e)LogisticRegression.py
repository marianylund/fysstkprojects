# See medium: https://medium.com/ai-in-plain-english/comparison-between-logistic-regression-and-neural-networks-in-classifying-digits-dc5e85cd93c3

# (Read more!) Logistic regression is the same as neural network with sigmoid but without hidden layers

from PROJECT_SETUP import ROJECT_ROOT_DIR
from nnreg.model import Model
from nnreg.trainer import Trainer
from nnreg.dataloader import DataLoader
from nnreg.config import Config

from nnreg.analysis_fun import plot_values_with_steps_and_info, get_max_value, show_heatmap, unpack, get_paths_of_results_where, plot_values_with_steps_and_info, train_save_configs, plot_lr_tran_val
from RegLib.load_save_data import get_best_dict, get_previous_checkpoint_as_dict, load_best_checkpoint, write_json, get_previous_checkpoints, load_data_as_dict

from time import time
from sklearn.model_selection import ParameterGrid
# For testing:
from sklearn.linear_model  import LogisticRegression

# For Analysis:
import numpy as np
from math import inf, isnan
import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

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

# Make sure that the configurations are fit for logisitc regression with MNIST
config_override = [
    'OPTIM.REGULARISATION', "l2",
    'OPTIM.BATCH_SIZE', 32,
    "MODEL.ACTIVATION_FUNCTIONS", ["softmax"],
    "MODEL.COST_FUNCTION", "ce",
    "DATA.NAME", "mnist",
    "MODEL.HIDDEN_LAYERS", [], # No hidden layers as it is regression
    "MODEL.EVAL_FUNC", "acc", # Compute accuracy
    "OUTPUT_DIR", "Teste)LogisticReg"
    ]

cfg = Config(config_override = config_override)
output_dir = ROJECT_ROOT_DIR.joinpath(cfg.OUTPUT_DIR)

data_loader = DataLoader(cfg)
train_save_configs(cfg, data_loader, output_dir)
best_data_dict = get_best_dict(output_dir)

data_loader = DataLoader(cfg, one_hot_encode=False)
test(cfg, data_loader, best_data_dict)

# plot_lr_tran_val(best_data_dict)






# ------------------------Parameter search-----------------------------------

# A bit different type of param search, without testing every time
def param_search(configs, output_dir:Path, param_grid:dict, train, test):

    start_time = time()
    param_grid = ParameterGrid(param_grid)
    total_combinations = len(param_grid)
    results = np.zeros(total_combinations)
    times = np.zeros((total_combinations, 2)) # min, sec
    print("Total combinations: ", total_combinations)

    # No changes in data, so dataloaders just here:
    cfg = Config(config_override = configs)
    data_loader = DataLoader(cfg)

    for i in range(total_combinations):
        val = param_grid[i]
        param = list(sum(val.items(), ())) # magic that flattens list of tuples

        name = "".join([str(i) for i in val.values()]).replace(".", "")
        print("Checking: ", param, " name: ", name)
        new_output_dir = output_dir.joinpath(name)

        ind_of_output = configs.index("OUTPUT_DIR")
        configs[ind_of_output + 1] = configs[ind_of_output] + "\\" + name

        new_cfg = Config(config_override = configs + param)

        train(new_cfg, data_loader, new_output_dir)
        best_data_dict = get_best_dict(new_output_dir)
        results[i] = best_data_dict["Test_eval"]
        times[i] = best_data_dict["Proccess_time"]

        print(f"\n{i + 1}/{total_combinations}. Time passed: {divmod(time() - start_time, 60)}\n")
        
    # Evaluate
    new_cfg = Config(config_override = configs)
    best_eval_i = 0
    if new_cfg.MODEL.EVAL_FUNC == "mse":
        best_eval_i = np.argmin(results)
    else:
        best_eval_i = np.argmax(results)

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

    data_loader_test = DataLoader(cfg, one_hot_encode=False)
    test(cfg, data_loader_test, {"Test_eval": results[best_eval_i]})

    print("Best eval: ", results[best_eval_i], " with param: ", param_grid[best_eval_i], ", time: ", times[best_eval_i])

param_grid = {
    'OPTIM.LR': [1e-3, 1e-2, 1e-1], 
    'MODEL.WEIGHT_INIT': ['random', 'he', 'xavier'],
    #'OPTIM.LR_DECAY': [0.0, 0.6, 0.9],
    'OPTIM.ALPHA': [0.1, 0.5, 0.9, 1.0],
}

#param_search(config_override, output_dir, param_grid, train, test)

# ------------------------Analysis of results-----------------------------------

def get_all_results_for_weight_init(path:Path):
    weight_inits = ['random', 'he', 'xavier']
    all_dir = [x for x in path.iterdir() if x.is_dir()]
    results = {}

    for w in weight_inits:
        results[w] = []
    
    for i in range(len(all_dir)):
        d = all_dir[i]
        cfg = Config(config_file = Path(d).joinpath("logistic_reg_mnist.yaml"))
        best = load_best_checkpoint(d)
        last_ckp = get_previous_checkpoints(d)[0]
        last = load_data_as_dict(Path(d).joinpath(last_ckp))
        new_val = list(last["Val_eval"].values())
        new_steps = list(map(int, last["Val_eval"].keys()))
        results[cfg.MODEL.WEIGHT_INIT].append({"WEIGHT_INIT": cfg.MODEL.WEIGHT_INIT, "LR": cfg.OPTIM.LR, "ALPHA": cfg.OPTIM.ALPHA, "Eval": best["Test_eval"],"Time": best["Proccess_time"], "Step": best["Step"], "Name": d})
        # "Val_eval": new_val, "Val_steps": new_steps,

    return results

def analyse_results(results, values_to_analyse = ("LR", "ALPHA"), round_up_to: float = 1, save_fig = False):
    min_val = get_max_value(results, "Eval") # MAX WHEN ACC AND MIN WHEN MSE
    print("Best val: ", min_val)
    best_checkpoint = load_best_checkpoint(min_val["Name"])
   
    cfg = Config(config_file = Path(min_val["Name"], "logistic_reg_mnist.yaml"))
    p = str(cfg.MODEL.WEIGHT_INIT)

    time_for_best_run = f'{min_val["Time"][0]:.0f} min {min_val["Time"][1]:.0f}'
    best_test_eval = f'{min_val["Eval"]:.5f}'
    
    # HEAT_MAP
    new_info = f'test score={best_test_eval}, time: {time_for_best_run}'
    info_to_add = {"Best result: ": new_info}
    s_results = unpack(results, values_to_unpack_on = values_to_analyse, replace_val_bigger = inf)
    position_index = s_results.index.get_loc(min_val[values_to_analyse[0]])
    position_column = s_results.columns.get_loc(min_val[values_to_analyse[1]])

    show_heatmap(s_results, info_to_add = info_to_add, patch_placement= (position_column, position_index), title = f"Logistic Regression {p}", xlabel = values_to_analyse[1], ylabel = values_to_analyse[0], show_bar = True, save_fig = save_fig)

# path_to_results = Path("Results", "e)LogisticReg")
# all_results = get_all_results_for_weight_init(path_to_results)

# plt.rcParams['font.size'] = 16 # To set the size of all plots to be bigger
# for res in all_results:
#     analyse_results(all_results[res], save_fig=True)


# values_to_plot = {}
# steps_to_plot = {}
# for res in all_results:
#     min_val = get_max_value(all_results[res], "Eval") # MAX WHEN ACC AND MIN WHEN MSE
#     print("Best val: ", min_val)

#     cfg = Config(config_file = Path(min_val["Name"], "logistic_reg_mnist.yaml"))

#     last = get_previous_checkpoint_as_dict(min_val["Name"])

#     new_key = f"{cfg.MODEL.WEIGHT_INIT}"#_{cfg.OPTIM.LR}_{cfg.OPTIM.ALPHA}"
#     values_to_plot[new_key] = list(last["Val_eval"].values())
#     steps_to_plot[new_key] = list(map(int, last["Val_eval"].keys()))

# plt.rcParams['font.size'] = 16 # To set the size of all plots to be bigger

# info_to_add = {}
# ylimit = (0.7, 1.0) #
# xlimit = None #(0, 50000) #
# save_fig = True
# plot_values_with_steps_and_info(steps_to_plot, values_to_plot, title = "Logistic Regression Best Runs (Zoomed)", xlimit = xlimit, ylabel = "Error",  info_to_add = info_to_add, ylimit = ylimit, save_fig = save_fig)

# best_runs_for_class = [Path('Results/e)LogisticReg/00105he'), ]