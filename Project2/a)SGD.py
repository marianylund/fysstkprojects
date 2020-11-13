from PROJECT_SETUP import ROJECT_ROOT_DIR
from nnreg.config import Config

from nnreg.dataloader import DataLoader
from nnreg.trainer import Trainer

from nnreg.analysis_fun import get_results_where, param_search, train_save_configs, plot_lr_tran_val,unpack, show_heatmap, get_min_value
from RegLib.load_save_data import write_json, load_best_checkpoint, get_previous_checkpoint_as_dict
from RegLib.HelperFunctions import get_best_dict, plot_values_with_info, plot_values_with_two_y_axis

# For testing:
from sklearn.linear_model import SGDRegressor

# For Analysis:
from math import inf
import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def test(cfg, data: DataLoader, best_data_dict):
    regr = SGDRegressor(loss = 'squared_loss',
                        average = False).fit(data.X_train, data.y_train)
    regr_test_pred = regr.predict(data.X_test)

    print("sklearn MLPRegressor: R2 : % .4f, MSE : % .4f" % (-Trainer.R2(data.X_test, data.y_test), Trainer.MSE(data.y_test.ravel(), regr_test_pred))) 
    print("Ours: R2 : % .4f, MSE : % .4f" % (best_data_dict["Test_r2"], best_data_dict["Test_eval"])) 


config_override = [
    "MODEL.ACTIVATION_FUNCTIONS", ["identity"], # to make it linear reg
    "MODEL.HIDDEN_LAYERS", [], # No layers
    "MODEL.EVAL_FUNC", "mse",
    "MODEL.COST_FUNCTION", "mse",
    "DATA.NAME", "franke",
    'DATA.FRANKIE.N', 1000,
    'DATA.FRANKIE.NOISE', 0.1,
    'OPTIM.LR', 1e-3, 
    'OPTIM.BATCH_SIZE', 60,
    "OPTIM.REGULARISATION", "none",
    "OUTPUT_DIR", "Testa)SDG",
]

cfg = Config(config_override = config_override)
    
output_dir = ROJECT_ROOT_DIR.joinpath(cfg.OUTPUT_DIR)

data_loader = DataLoader(cfg)
train_save_configs(cfg, data_loader, output_dir)
best_data_dict = get_best_dict(output_dir)
test(cfg, data_loader, best_data_dict)
plot_lr_tran_val(best_data_dict, y1_label = "Error", info_to_add = {}, title = "SGD", ylimit = None, save_fig = False)


# ------------------------Parameter search-----------------------------------

param_grid = {
    'OPTIM.LR': [1e-3, 1e-4], 
    'OPTIM.BATCH_SIZE': [60], # try with 60 ?
    'OPTIM.LR_DECAY': [0.0, 0.6, 0.9],
    'OPTIM.ALPHA': [0.3, 0.5, 0.9, 1.0],
    'DATA.FRANKIE.P': [5],
    'OPTIM.REGULARISATION': ["l1"],
}

#param_search(config_override, output_dir, param_grid, train, test)

# ------------------------Analysis of results-----------------------------------

def get_all_results_for_p(path:Path):
    polynomials = [2, 5, 10, 15]
    all_dir = [x for x in path.iterdir() if x.is_dir()]
    results = {}
    for pol in polynomials:
        results[pol] = []

    for i in range(len(all_dir)):
        d = all_dir[i]
        cfg = Config(config_file = Path(d).joinpath("sgd.yaml"))
        best = load_best_checkpoint(d)

        results[cfg.DATA.FRANKIE.P].append({"LR": cfg.OPTIM.LR, "batch_size": cfg.OPTIM.BATCH_SIZE, "Eval": best["Test_eval"],"Time": best["Proccess_time"], "Step": best["Step"], "Name": d})

    return results

def analyse_results(results, round_up_to: float = 1, save_fig = False):
    min_val = get_min_value(results, "Eval")
    print("Best val: ", min_val)
    best_checkpoint = load_best_checkpoint(min_val["Name"])
   
    cfg = Config(config_file = Path(min_val["Name"], "sgd.yaml"))
    p = str(cfg.DATA.FRANKIE.P)

    time_for_best_run = f'{min_val["Time"][0]:.0f} min {min_val["Time"][1]:.0f}'
    best_test_eval = f'{min_val["Eval"]:.5f}'
    
    # HEAT_MAP
    info_to_add = {}
    s_results = unpack(results, replace_val_bigger = inf)
    position_index = s_results.index.get_loc(min_val["batch_size"])
    position_column = s_results.columns.get_loc(min_val["LR"])

    show_heatmap(s_results, info_to_add = info_to_add, patch_placement= (position_column, position_index), title = f"SGD on Franke p={p}", xlabel = 'Learning rate', ylabel = 'Batch size', show_bar = False, save_fig = save_fig)

    print(f'{min_val["Eval"]} replacing with: {round_up_to}')
    s_results = unpack(results, replace_val_bigger = round_up_to)
    show_heatmap(s_results, info_to_add = info_to_add, patch_placement= (position_column, position_index), title = f"SGD on Franke p={p} (Filtered)", xlabel = 'Learning rate', ylabel = 'Batch size', show_bar = True, save_fig = save_fig)

    new_info = f'={p}, test score={best_test_eval}, time: {time_for_best_run}'
    # PLOTS
    info_to_add = {
        "p": new_info,
        "File name: ": str(min_val["Name"]).replace("\\", "_"),
    }
    print(info_to_add)
    plot_lr_tran_val(best_checkpoint, ylimit = (0.0, 0.1), title = f'Best Run Zoomed In p={p}', info_to_add = info_to_add, save_fig = save_fig)
    plot_lr_tran_val(best_checkpoint,  ylimit = (0.0, 1.0), title = f'Best Run p={p}', info_to_add = info_to_add, save_fig = save_fig)

# Creating heatmaps and plot of best result for polynomials
# path_to_results = Path("Results").joinpath("SDG")
# results = get_all_results_for_p(path_to_results)

# plt.rcParams['font.size'] = 16 # To set the size of all plots to be bigger
# for p in results:
#     analyse_results(results[p], round_up_to = 1, save_fig = True)

# Concentrating on polynomial 5

#path_to_results = Path("Results").joinpath("SGD_Ridge")
def get_ridge_results(path:Path):
    all_dir = [x for x in path.iterdir() if x.is_dir()]
    results = []

    for i in range(len(all_dir)):
        d = all_dir[i]
        cfg = Config(config_file = Path(d).joinpath("sgd.yaml"))
        best = load_best_checkpoint(d)

        results.append({"LR": cfg.OPTIM.LR, "LR_DECAY": cfg.OPTIM.LR_DECAY, "ALPHA": cfg.OPTIM.ALPHA, "Eval": best["Test_eval"],"Time": best["Proccess_time"], "Step": best["Step"], "Name": d})
    return results


def analyse_ridge_results(results, values_to_unpack_on = ("LR_DECAY", "ALPHA"), round_up_to: float = 1, save_fig = False):
    min_val = get_min_value(results, "Eval")
    print("Best val: ", min_val)
    best_checkpoint = load_best_checkpoint(min_val["Name"])
   
    cfg = Config(config_file = Path(min_val["Name"], "sgd.yaml"))
    p = str(cfg.DATA.FRANKIE.P)

    time_for_best_run = f'{min_val["Time"][0]:.0f} min {min_val["Time"][1]:.0f}'
    best_test_eval = f'{min_val["Eval"]:.5f}'
    
    # HEAT_MAP
    info_to_add = {}
    s_results = unpack(results, values_to_unpack_on = values_to_unpack_on, replace_val_bigger = inf)
    position_index = s_results.index.get_loc(min_val[values_to_unpack_on[0]])
    position_column = s_results.columns.get_loc(min_val[values_to_unpack_on[1]])

    show_heatmap(s_results, info_to_add = info_to_add, patch_placement= (position_column, position_index), title = f"Ridge SGD on Franke p={p}", xlabel = 'ALPHA', ylabel = 'LR_DECAY', show_bar = False, save_fig = save_fig)

    new_info = f'test score={best_test_eval}, time: {time_for_best_run}'
    # PLOTS
    info_to_add = {
        "Info: ": new_info,
        "File name: ": str(min_val["Name"]).replace("\\", "_"),
    }
    print(info_to_add)
    #plot_lr_tran_val(best_checkpoint, ylimit = (0.0, 0.1), title = f'Best Run Zoomed In p={p}', info_to_add = info_to_add, save_fig = save_fig)
    #plot_lr_tran_val(best_checkpoint,  ylimit = (0.0, 1.0), title = f'Best Run p={p}', info_to_add = info_to_add, save_fig = save_fig)

#ridge_results = get_results_where(get_ridge_results(path_to_results), "LR", 1e-3) 
#analyse_ridge_results(ridge_results, save_fig = True)