# Useful info: https://towardsdatascience.com/implementing-different-activation-functions-and-weight-initialization-methods-using-python-c78643b9f20f

from nnreg.model import Model
from nnreg.trainer import Trainer
from nnreg.dataloader import DataLoader
from RegLib.HelperFunctions import create_frankie_data, create_X, plot_values_with_info,plot_values_with_two_y_axis
from nnreg.config import Config
from PROJECT_SETUP import ROJECT_ROOT_DIR
from RegLib.load_save_data import load_best_checkpoint, write_json, get_previous_checkpoints, load_data_as_dict
from nnreg.analysis_fun import show_heatmap, get_min_value, unpack, get_paths_of_results_where, plot_values_with_steps_and_info, param_search, train_save_configs, plot_lr_tran_val

from sklearn.model_selection import ParameterGrid
# For testing:
from sklearn.neural_network import MLPRegressor

# For Analysis:
from math import inf, isnan
import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Compare to sklearn, is there a better function to compare to?:
def test(cfg, data: DataLoader, best_data_dict):
    regr = MLPRegressor(hidden_layer_sizes = cfg.MODEL.HIDDEN_LAYERS,
                        activation = "identity",
                        solver = "sgd",
                        early_stopping=True).fit(data.X_train, data.y_train.ravel())

    regr_test_pred = regr.predict(data.X_test)

    print("sklearn: R2 : % .4f, MSE : % .4f" % (Trainer.R2(data.y_test.ravel(), regr_test_pred), Trainer.MSE(data.y_test.ravel(), regr_test_pred))) 
    print("Ours: R2 : % .4f, MSE : % .4f" % (best_data_dict["Test_r2"], best_data_dict["Test_eval"])) 


config_override = [
    "MODEL.ACTIVATION_FUNCTIONS", ["sigmoid", "identity"], # will need to vary activations in c)
    "MODEL.HIDDEN_LAYERS", [5],  # this will need to vary
    "MODEL.WEIGHT_INIT", "xavier", # this will need to vary in c) # {'random', 'he', 'xavier', 'zeros'}
    "MODEL.EVAL_FUNC", "mse",
    "MODEL.COST_FUNCTION", "mse",
    "OPTIM.REGULARISATION", "none",
    "OPTIM.BATCH_SIZE", 60,
    "OPTIM.LR", 1e-3, # for now just concentrate on this lr
    "DATA.NAME", "franke",
    'DATA.FRANKIE.P', 5,
    'DATA.FRANKIE.N', 1000,
    'DATA.FRANKIE.NOISE', 0.1,
    "OUTPUT_DIR", "Testc)ActFun",
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
    'MODEL.WEIGHT_INIT': ['random', 'he', 'xavier', 'zeros'], 
    "MODEL.ACTIVATION_FUNCTIONS": [["sigmoid", "identity"], ["tanh", "identity"], ["relu", "identity"]],
}

#param_search(config_override, output_dir, param_grid, train, test)

param_grid = {
    'MODEL.WEIGHT_INIT': ['random', 'he', 'xavier', 'zeros'], 
    "MODEL.ACTIVATION_FUNCTIONS": [["leaky_relu", "identity"]],
    "MODEL.LEAKY_SLOPE": [-0.1, 0.1] # Maybe 0.01 is better?
}

#param_search(config_override, output_dir, param_grid, train, test)


# ------------------------Analysis of results-----------------------------------

def get_all_results_for_weight_init(path:Path, leaky = False):
    weight_inits = ['random', 'he', 'xavier', 'zeros']
    all_dir = [x for x in path.iterdir() if x.is_dir()]
    results = []

    for i in range(len(all_dir)):
        d = all_dir[i]
        cfg = Config(config_file = Path(d).joinpath("multilayer_model.yaml"))
        if (leaky and cfg.MODEL.ACTIVATION_FUNCTIONS[0] == "leaky_relu") or (not leaky and cfg.MODEL.ACTIVATION_FUNCTIONS[0] != "leaky_relu"):
            best = load_best_checkpoint(d)
            last_ckp = get_previous_checkpoints(d)[0]
            last = load_data_as_dict(Path(d).joinpath(last_ckp))
            new_val = list(last["Val_eval"].values())
            new_steps = list(map(int, last["Val_eval"].keys()))
            results.append({"WEIGHT_INIT": cfg.MODEL.WEIGHT_INIT, "ACTIVATION": cfg.MODEL.ACTIVATION_FUNCTIONS[0], "LEAKY_SLOPE": cfg.MODEL.LEAKY_SLOPE, "Eval": best["Test_eval"],"Time": best["Proccess_time"], "Step": best["Step"], "Val_eval": new_val, "Val_steps": new_steps, "Name": d})

    return results


def analyse_results(results, values_to_analyse = ("LR_DECAY", "LR"), round_up_to: float = 1, save_fig = False):
    min_val = get_min_value(results, "Eval") # MAX WHEN ACC AND MIN WHEN MSE
    print("Best val: ", min_val)
    best_checkpoint = load_best_checkpoint(min_val["Name"])
   
    cfg = Config(config_file = Path(min_val["Name"], "multilayer_model.yaml"))
    p = str(cfg.MODEL.WEIGHT_INIT)

    time_for_best_run = f'{min_val["Time"][0]:.0f} min {min_val["Time"][1]:.0f}'
    best_test_eval = f'{min_val["Eval"]:.5f}'
    
    # HEAT_MAP
    info_to_add = {}
    s_results = unpack(results, values_to_unpack_on = values_to_analyse, replace_val_bigger = inf)
    position_index = s_results.index.get_loc(min_val[values_to_analyse[0]])
    position_column = s_results.columns.get_loc(min_val[values_to_analyse[1]])

    show_heatmap(s_results, info_to_add = info_to_add, patch_placement= (position_column, position_index), title = f"Franke NN", xlabel = values_to_analyse[1], ylabel = values_to_analyse[0], show_bar = True, save_fig = save_fig)

    new_info = f'test score={best_test_eval}, time: {time_for_best_run}'
    # PLOTS
    info_to_add = {
        "Results: ": new_info,
        "File name: ": str(min_val["Name"]).replace("\\", "_"),
    }
    print(info_to_add)
    #plot_lr_tran_val(best_checkpoint, y1_label = "Error", title = f'Best Run Weight init = {p}', info_to_add = info_to_add, save_fig = save_fig)

#path_to_results = Path("Results", "SimpleNN")



#all_results_with_leaky = get_all_results_for_weight_init(path_to_results, leaky=True)
#analyse_results(all_results_with_leaky, values_to_analyse = ("LEAKY_SLOPE", "WEIGHT_INIT"))

def analyse_without_leaky():
    all_results_without_leaky = get_all_results_for_weight_init(path_to_results)
    analyse_results(all_results_without_leaky, values_to_analyse = ("ACTIVATION", "WEIGHT_INIT"), save_fig = True)
    values_to_plot = {}
    steps_to_plot = {}
    clean_of_exploding = True
    for result in all_results_without_leaky:
        new_val = result["Val_eval"]
        weight_init = result["WEIGHT_INIT"]
        act = result["ACTIVATION"]
        if(not clean_of_exploding or (new_val[-1] != inf and not isnan(new_val[-1]))):
            new_key = f"{weight_init}_{act}"
            values_to_plot[new_key] = new_val
            steps_to_plot[new_key] = result["Val_steps"]

    info_to_add = {}
    ylimit = (0.01, 0.04) #
    xlimit = None #(0, 50000) #
    save_fig = False

    #test(cfg, data_loader, best_checkpoint_10_5)
    #plot_values_with_steps_and_info(steps_to_plot, values_to_plot, title = "Weight Init and Activations on Franke", xlimit = xlimit, ylabel = "Error",  info_to_add = info_to_add, ylimit = ylimit, save_fig = save_fig)


def analyse_with_leaky():
    save_fig = True
    all_results_with_leaky = get_all_results_for_weight_init(path_to_results, leaky=True)
    analyse_results(all_results_with_leaky, values_to_analyse = ("LEAKY_SLOPE", "WEIGHT_INIT"), save_fig = save_fig)
    
    values_to_plot = {}
    steps_to_plot = {}
    clean_of_exploding = True
    for result in all_results_with_leaky:
        new_val = result["Val_eval"]
        weight_init = result["WEIGHT_INIT"]
        act = result["LEAKY_SLOPE"]
        if(not clean_of_exploding or (new_val[-1] != inf and not isnan(new_val[-1]))):
            new_key = f"{weight_init}{act}"
            values_to_plot[new_key] = new_val
            steps_to_plot[new_key] = result["Val_steps"]

    info_to_add = {}
    ylimit = (0.01, 0.04) #
    xlimit = None #(0, 50000) #

    #test(cfg, data_loader, best_checkpoint_10_5)
    plot_values_with_steps_and_info(steps_to_plot, values_to_plot, title = "Weight Init with leaky ReLU on Franke", xlimit = xlimit, ylabel = "Error",  info_to_add = info_to_add, ylimit = ylimit, save_fig = save_fig)

#analyse_without_leaky()
#analyse_with_leaky()