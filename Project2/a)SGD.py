from NeuralNetwork.SGD import SGD as SGD
from RegLib.HelperFunctions import create_frankie_data, create_X, plot_values_with_info
from NeuralNetwork.config import Config
from PROJECT_SETUP import ROJECT_ROOT_DIR
from RegLib.load_save_data import load_best_checkpoint

# For testing:
from sklearn.neural_network import MLPRegressor



cfg = Config()

output_dir = ROJECT_ROOT_DIR.joinpath(cfg.OUTPUT_DIR)
# Save settings in the same folder as checkpoints
def train():
    cfg.dump(output_dir.joinpath("sdg.yaml"))
#cfg = Config(config_file = path_of_cfg, config_override = ["OPTIM.NUM_EPOCHS", 500])

    x, y, z = create_frankie_data(cfg.SEED, cfg.DATA.FRANKIE.N, cfg.DATA.FRANKIE.NOISE)
    X = create_X(x, y, cfg.DATA.FRANKIE.P)
    return SGD().train_and_test(cfg = cfg, X = X, y=z, checkpoints_path = output_dir)

sgd = train()

best_data_dict = load_best_checkpoint(output_dir)

# Compare to sklearn:
regr = MLPRegressor(early_stopping=True).fit(sgd.X_train, sgd.y_train.ravel())
regr_test_pred = regr.predict(sgd.X_test)
print("sklearn: R2 : % .4f, MSE : % .4f" % (sgd.R2(sgd.X_test, sgd.y_test), sgd.MSE(sgd.y_test.ravel(), regr_test_pred))) 
print("Ours: R2 : % .4f, MSE : % .4f" % (best_data_dict["Test_r2"], best_data_dict["Test_mse"])) 


values_to_plot = {
    "Train_mse": list(best_data_dict["Train_mse"].values()),
    #"Train_r2": list(best_data_dict["Train_r2"].values()),
    "Learning_rate": list(best_data_dict["Learning_rate"].values()),
}
epochs = [x for x in range(best_data_dict["Step"] + 1)]
#plot_values_with_info(epochs, values_to_plot, title = "SGD", xlabel = "Steps", save_fig = False)







