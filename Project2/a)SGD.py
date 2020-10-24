from NeuralNetwork.SGD import SGD as SGD
from RegLib.HelperFunctions import create_frankie_data, create_X
from NeuralNetwork.config import Config
from PROJECT_SETUP import ROJECT_ROOT_DIR

cfg = Config()

output_dir = ROJECT_ROOT_DIR.joinpath(cfg.OUTPUT_DIR)
# Save settings in the same folder as checkpoints
cfg.dump(output_dir.joinpath("sdg.yaml"))
#cfg = Config(config_file = path_of_cfg, config_override = ["OPTIM.NUM_EPOCHS", 500])

x, y, z = create_frankie_data(cfg.SEED, cfg.DATA.FRANKIE.N, cfg.DATA.FRANKIE.NOISE)
X = create_X(x, y, cfg.DATA.FRANKIE.P)
sgd = SGD().train_and_test(cfg = cfg, X = X, y=z, checkpoints_path = output_dir)




