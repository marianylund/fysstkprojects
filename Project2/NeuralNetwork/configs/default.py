# Inspired by SSD300 from https://github.com/lufficc/SSD
from yacs.config import CfgNode as CN

cfg = CN()

# train configs
cfg.NUM_EPOCHS = 50
cfg.BATCH_SIZE = 32
cfg.LR = 1e-3
cfg.L2_REG_LAMBDA = 0.001

# test options
cfg.EVAL_STEP = 500 # Evaluate dataset every eval_step, disabled when eval_step < 0
cfg.MODEL_SAVE_STEP = 500 # Save checkpoint every save_step
cfg.LOG_STEP = 10 # Print logs every log_stepPrint logs every log_step
cfg.OUTPUT_DIR = "outputs"
cfg.DATASET_DIR = "datasets"