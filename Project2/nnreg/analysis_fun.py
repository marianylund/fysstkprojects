import numpy as np
import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from RegLib.HelperFunctions import save_figure, plot_values_with_two_y_axis
from matplotlib.patches import Rectangle

def get_min_value(results_to_get_min_from, value:str):
    min_ind = get_min_value_index(results_to_get_min_from, value)
    return results_to_get_min_from[min_ind]

def get_min_value_index(results_to_get_min_from, value:str) -> int:
    evals = map(lambda x: x[value], results_to_get_min_from)
    return np.asarray(list(evals)).argmin()

def get_list(results_to_get_from, value:str):
    evals = map(lambda x: x[value], results_to_get_from)
    return list(evals)

def get_list_of_tuples(results_to_get_from, value1:str, value2:str):
    evals = map(lambda x: (x[value1], x[value2]), results_to_get_from)
    return list(evals)

def replace_if_bigger(list_with_val, max_val, replace_with = None):
    return [replace_with if x > max_val else x for x in list_with_val]

def unpack(res, replace_val_bigger = 0.026203, replace_with = None):
    evals = replace_if_bigger(get_list(res, "Eval"), replace_val_bigger, replace_with)
    print("Max value: ", max(filter(lambda v: v is not None, evals)))
    tuples_data = get_list_of_tuples(res, "batch_size", "LR")
    index = pd.MultiIndex.from_tuples(tuples_data)
    s_results = pd.Series(evals, index=index)
    s_results = s_results.unstack(level=-1)
    return s_results

def show_heatmap(data, title, xlabel, ylabel, patch_placement = None, show_bar = True, save_fig = False):
    cmap = sb.diverging_palette(0, 230, 90, 60, as_cmap=True)
    fig, ax = plt.subplots(figsize=(7, 5))

    sb.heatmap(data, annot=True, linewidth=0.3, cbar = show_bar, cmap=cmap, square=True)  #  annot=True, fmt=".2f", 
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if patch_placement != None:
        ax.add_patch(Rectangle(patch_placement,1,1, fill=False, edgecolor='green', lw=3))

    b, t = plt.ylim() # discover the values for bottom and top
    b += 0.5 # Add 0.5 to the bottom
    t -= 0.5 # Subtract 0.5 from the top
    plt.ylim(b, t) # update the ylim(bottom, top) values
    plt.title(title, loc='left', fontsize=12, fontweight=0)

    if save_fig:
        save_figure(title)
    else: 
        plt.show()

    plt.show()

def plot_lr_tran_val(best_data_dict, info_to_add = {}, ylimit = None, save_fig = False):
    values_to_plot = {
        "Train_mse": list(best_data_dict["Train_eval"].values()),
        "Val_mse": list(best_data_dict["Val_eval"].values()),
        #"Train_r2": list(best_data_dict["Train_r2"].values()),
    }
    y2 = { "Learning_rate": list(best_data_dict["Learning_rate"].values())}

    steps = list(best_data_dict["Train_eval"].keys())
    plot_values_with_two_y_axis(steps, values_to_plot, y2, y1_label = "Error", title = "SGD", info_to_add = info_to_add, ylimit = ylimit, save_fig = save_fig)