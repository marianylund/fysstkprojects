from nnreg.SGD import SGD as SGD
from RegLib.HelperFunctions import create_frankie_data, create_X, plot_values_with_info,plot_values_with_two_y_axis
from nnreg.config import Config
from PROJECT_SETUP import ROJECT_ROOT_DIR
from RegLib.load_save_data import load_best_checkpoint, write_json

import numpy as np
from sklearn.model_selection import ParameterGrid
# For testing:
from sklearn.neural_network import MLPRegressor

cfg = Config()
output_dir = ROJECT_ROOT_DIR.joinpath(cfg.OUTPUT_DIR)

def train(cfg, output_dir):
    cfg.dump(output_dir.joinpath("sdg.yaml"))
    #cfg = Config(config_file = path_of_cfg, config_override = ["OPTIM.NUM_EPOCHS", 500])

    x, y, z = create_frankie_data(cfg.SEED, cfg.DATA.FRANKIE.N, cfg.DATA.FRANKIE.NOISE)
    X = create_X(x, y, cfg.DATA.FRANKIE.P)
    return SGD().train_and_test(cfg = cfg, X = X, y=z, checkpoints_path = output_dir)

# Compare to sklearn, is there a better function to compare to?:
def test(sgd, best_data_dict):
    regr = MLPRegressor(early_stopping=True).fit(sgd.X_train, sgd.y_train.ravel())
    regr_test_pred = regr.predict(sgd.X_test)
    print("sklearn: R2 : % .4f, MSE : % .4f" % (sgd.R2(sgd.X_test, sgd.y_test), sgd.MSE(sgd.y_test.ravel(), regr_test_pred))) 
    print("Ours: R2 : % .4f, MSE : % .4f" % (best_data_dict["Test_r2"], best_data_dict["Test_mse"])) 

def plot(best_data_dict):
    values_to_plot = {
        "Train_mse": list(best_data_dict["Train_mse"].values())
        #"Train_r2": list(best_data_dict["Train_r2"].values()),
    }
    y2 = { "Learning_rate": list(best_data_dict["Learning_rate"].values())}

    steps = [x for x in range(best_data_dict["Step"] + 1)]
    plot_values_with_two_y_axis(steps, values_to_plot, y2, title = "SGD", save_fig = False)

def param_search(output_dir):

    param_grid = ParameterGrid({
        'OPTIM.LR': [1e-1, 1e-2, 1e-3, 1e-4], 
        'OPTIM.EARLY_STOP_LR_STEP': [-1.0, 1e-5, 1e-8, 1e-10]})

    results = np.zeros(len(param_grid))
    times = np.zeros(len(param_grid))

    for i in range(len(param_grid)):
        val = param_grid[i]
        param = list(sum(val.items(), ())) # magic that flattens list of tuples
        new_cfg = Config(config_override = param)
        name = "".join([str(i) for i in val.values()]).replace(".", "")
        print("Checking: ", param, " name: ", name)
        new_output_dir = output_dir.joinpath(name)
        sgd = train(new_cfg, new_output_dir)
        results[i] = sgd.best_test_mse
        times[i] = sgd.process_time
        print("MSE: ", results[i], ", time: ", times[i])
    
    best_mse_i = np.argmin(results)
    results_dict = {
        "best_index":  best_mse_i,
        "best_mse": results[best_mse_i],
        "best_param": param_grid[best_mse_i],
        "best_time": times[best_mse_i],
        "param" : param_grid,
        "results": results,
        "times": times
    }
    write_json(results_dict, output_dir.joinpath("param_search_results.json"))
    print("Best mse: ", results[best_mse_i], " with param: ", param_grid[best_mse_i], ", time: ", times[best_mse_i])
    




#sgd = train(cfg, output_dir)
#best_data_dict = load_best_checkpoint(output_dir)
#test(sgd, best_data_dict)
#plot(best_data_dict)
#print(list(best_data_dict["Learning_rate"].values()))
param_search(output_dir)





