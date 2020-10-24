from NeuralNetwork.SGD import SGD as SGD
from RegLib.HelperFunctions import create_frankie_data, create_X, plot_values_with_info,plot_values_with_two_y_axis
from NeuralNetwork.config import Config
from PROJECT_SETUP import ROJECT_ROOT_DIR
from RegLib.load_save_data import load_best_checkpoint

# For testing:
from sklearn.neural_network import MLPRegressor

cfg = Config()
output_dir = ROJECT_ROOT_DIR.joinpath(cfg.OUTPUT_DIR)

def train():
    cfg.dump(output_dir.joinpath("sdg.yaml"))
    #cfg = Config(config_file = path_of_cfg, config_override = ["OPTIM.NUM_EPOCHS", 500])

    x, y, z = create_frankie_data(cfg.SEED, cfg.DATA.FRANKIE.N, cfg.DATA.FRANKIE.NOISE)
    X = create_X(x, y, cfg.DATA.FRANKIE.P)
    return SGD().train_and_test(cfg = cfg, X = X, y=z, checkpoints_path = output_dir)

# Compare to sklearn:
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

#sgd = train()
best_data_dict = load_best_checkpoint(output_dir)
#test(sgd, best_data_dict)
plot(best_data_dict)
#print(list(best_data_dict["Learning_rate"].values()))





