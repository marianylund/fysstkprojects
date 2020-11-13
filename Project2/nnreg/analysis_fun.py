import numpy as np
import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from time import time
from nnreg.config import Config
from nnreg.dataloader import DataLoader
from nnreg.trainer import Trainer

from RegLib.HelperFunctions import get_best_dict, save_figure, plot_values_with_two_y_axis, parse_info_for_plot
from RegLib.load_save_data import write_json
from sklearn.model_selection import ParameterGrid
from matplotlib.patches import Rectangle

from yacs.config import CfgNode as CN

def get_min_value(results_to_get_min_from, value:str):
    min_ind = get_min_value_index(results_to_get_min_from, value)
    return results_to_get_min_from[min_ind]

def get_min_value_index(results_to_get_min_from, value:str) -> int:
    evals = map(lambda x: x[value], results_to_get_min_from)
    return np.asarray(list(evals)).argmin()

def get_max_value(results_to_get_max_from, value:str):
    max_ind = get_max_value_index(results_to_get_max_from, value)
    return results_to_get_max_from[max_ind]

def get_max_value_index(results_to_get_max_from, value:str) -> int:
    evals = map(lambda x: x[value], results_to_get_max_from)
    return np.asarray(list(evals)).argmax()

def get_list(results_to_get_from, value:str):
    evals = map(lambda x: x[value], results_to_get_from)
    return list(evals)

def get_results_where(results_to_get_from, value:str, value_equals):
    evals = filter(lambda x: x if x[value] == value_equals else None, results_to_get_from)
    return list(evals)

def get_paths_of_results_where(results_to_get_from, value:str, value_equals):
    evals = filter(lambda x: x if x[value] == value_equals else None, results_to_get_from)
    new_evals = map(lambda x: Path(x["Name"]), evals)
    return list(new_evals)

def get_list_of_tuples(results_to_get_from, value1:str, value2:str):
    evals = map(lambda x: (x[value1], x[value2]), results_to_get_from)
    return list(evals)

def replace_if_bigger(list_with_val, max_val, replace_with = None):
    return [replace_with if x > max_val else x for x in list_with_val]

def unpack(res, values_to_unpack_on = ("batch_size", "LR"), replace_val_bigger = 0.026203, replace_with = None):
    evals = replace_if_bigger(get_list(res, "Eval"), replace_val_bigger, replace_with)
    tuples_data = get_list_of_tuples(res, values_to_unpack_on[0], values_to_unpack_on[1])
    index = pd.MultiIndex.from_tuples(tuples_data)
    s_results = pd.Series(evals, index=index)
    s_results = s_results.unstack(level=-1)
    return s_results

def show_heatmap(data, title, xlabel, ylabel, info_to_add = {}, patch_placement = None, show_bar = True, save_fig = False):
    cmap = sb.diverging_palette(0, 230, 90, 60, as_cmap=True)
    fig, ax = plt.subplots(figsize=(7, 5))

    sb.heatmap(data, annot=True, linewidth=0.3, cbar = show_bar, cmap=cmap, square=True, annot_kws={'size':14})  #  annot=True, fmt=".2f", 
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if patch_placement != None:
        ax.add_patch(Rectangle(patch_placement,1,1, fill=False, edgecolor='green', lw=3))

    b, t = plt.ylim() # discover the values for bottom and top
    b += 0.5 # Add 0.5 to the bottom
    t -= 0.5 # Subtract 0.5 from the top
    plt.ylim(b, t) # update the ylim(bottom, top) values
    plt.title(title, loc='left', fontsize=18, fontweight=0)
    
    info_str, title_info = parse_info_for_plot(info_to_add)
    
    if info_str != "":
        plt.figtext(0.1, -0.1, info_str, ha="left", fontsize=12)

    if save_fig:
        save_figure(title)# + title_info)
        plt.cla()
    else: 
        plt.show()

def train_save_configs(cfg, data: DataLoader, output_dir):
    cfg.dump(output_dir.joinpath("configs.yaml")) #sgd
    return Trainer().train_and_save(cfg = cfg, data_loader = data, checkpoints_path = output_dir)

def plot_lr_tran_val(best_data_dict, y1_label = "Error", info_to_add = {}, title = "SGD", ylimit = None, save_fig = False):
    values_to_plot = {
        "Train": list(best_data_dict["Train_eval"].values()),
        "Val": list(best_data_dict["Val_eval"].values()),
        #"Train_r2": list(best_data_dict["Train_r2"].values()),
    }
    y2 = { "Learning_rate": list(best_data_dict["Learning_rate"].values())}

    steps = list(map(int, best_data_dict["Train_eval"].keys()))
    plot_values_with_two_y_axis(steps, values_to_plot, y2, y1_label = y1_label, title = title, info_to_add = info_to_add, ylimit = ylimit, save_fig = save_fig)

def param_search(configs:CN, output_dir:Path, param_grid:dict, train, test):

    start_time = time()
    param_grid = ParameterGrid(param_grid)
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
    print("Best eval: ", results[best_eval_i], " with param: ", param_grid[best_eval_i], ", time: ", times[best_eval_i])


def plot_values_with_steps_and_info(steps, values_to_plot, title = "", xlabel = "Steps", xlimit= None, ylimit= None,  ylabel = "Error", info_to_add = {}, xscale = "linear", save_fig = False):
    plt.style.use('seaborn-darkgrid')
    fig, ax = plt.subplots()
    for key in values_to_plot:
        ax.plot(steps[key], values_to_plot[key], label=key)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    info_str, title_info = parse_info_for_plot(info_to_add)
    
    if info_str != "":
        plt.figtext(0.1, -0.1, info_str, ha="left", fontsize=7)

    if ylimit != None:
        ax.set_ylim(ylimit[0], ylimit[1])
    if xlimit != None:
        ax.set_xlim(xlimit[0], xlimit[1])
    
    plt.title(title, loc='left', fontsize=18, fontweight=0)
    plt.xscale(xscale)
    plt.tight_layout()
    if save_fig:
        save_figure(title + title_info)
    else:
        plt.show()
        print(info_str)
