# Inspired by SSD300 from https://github.com/lufficc/SSD
# and https://github.com/nocaps-org/updown-baseline/blob/266081042bc2f0c3e6676ee0b5e204036cb6a74a/updown/config.py/

from yacs.config import CfgNode as CN
from pathlib import Path
from typing import Any, List, Optional

class Config(object):
    r"""
    This class provides package-wide configuration management. It is a nested dict-like structure
    with nested keys accessible as attributes. It contains sensible default values, which can be
    modified by (first) a YAML file and (second) a list of attributes and values.
    Extended Summary
    ----------------
    This class definition contains default hyperparameters for the UpDown baseline from our paper.
    Modification of any parameter after instantiating this class is not possible, so you must
    override required parameter values in either through ``config_file`` or ``config_override``.
    Parameters
    ----------
    config_file: str
        Path to a YAML file containing configuration parameters to override.
    config_override: List[Any], optional (default= [])
        A list of sequential attributes and values of parameters to override. This happens after
        overriding from YAML file.
    Examples
    --------
    Let a YAML file named "config.yaml" specify these parameters to override::
        RANDOM_SEED: 42
        OPTIM:
          BATCH_SIZE: 512
    >>> _C = Config("config.yaml", ["OPTIM.BATCH_SIZE", 2048])
    >>> _C.RANDOM_SEED  # default: 0
    42
    >>> _C.OPTIM.BATCH_SIZE  # default: 150
    2048"""
    
    def __init__(self, config_file: Optional[Path] = None, config_override: List[Any] = []):
        _cfg = CN()
        _cfg.SEED = 2018
        _cfg.SAVE_FIG = False
        _cfg.TEST_SIZE = 0.2
        _cfg.SHUFFLE = True

        _cfg.OPTIM = CN()
        _cfg.OPTIM.NUM_EPOCHS = 1
        _cfg.OPTIM.BATCH_SIZE = 20
        _cfg.OPTIM.LR = 1e-3
        _cfg.OPTIM.L2_REG_LAMBDA = 1.0 # 1.0 to turn it off
        _cfg.OPTIM.EARLY_STOP_LR_STEP = -1.0 #1e-5 # -1 to turn it off
        _cfg.OPTIM.LR_DECAY = 0.0 # 0 to turn it off
        _cfg.OPTIM.USE_MOMENTUM = False 
        _cfg.OPTIM.MOMENTUM = 0.9 # if 1 then no friction, used only if USE_MOMENTUM is true

        _cfg.MODEL = CN()
        _cfg.MODEL.HIDDEN_LAYERS = [10] # the last layer will be added based on the data
        _cfg.MODEL.ACTIVATION_FUNCTIONS = ["sigmoid", "identity"] # {'identity', 'sigmoid', 'tanh', 'relu', 'leaky_relu'}
        _cfg.MODEL.LEAKY_SLOPE = 0.1 # Is used only if activation function is "leaky_relu"
        _cfg.MODEL.COST_FUNCTION = "mse" # {'mse', 'ce'}
        _cfg.MODEL.WEIGHT_INIT = "random" # {'random', 'he', 'xavier', 'zeros'}

        # scheduler: Callable[[float, float, float], float] = None

        # test options
        #_cfg.EVAL_STEP = -1 # Evaluate dataset every eval_step, disabled when eval_step < 0
        _cfg.MODEL_SAVE_STEP = 500 # Save checkpoint every save_step
        #_cfg.LOG_STEP = 10 # Print logs every log_stepPrint logs every log_step

        _cfg.OUTPUT_DIR = "test_data_loader"  # folder inside checkpoints

        _cfg.DATA = CN()
        _cfg.DATA.NAME = "franke" # {"franke", "mnist"}
        # Settings for Frankie data
        _cfg.DATA.FRANKIE = CN()
        _cfg.DATA.FRANKIE.N = 100
        _cfg.DATA.FRANKIE.NOISE = 0.1
        _cfg.DATA.FRANKIE.P = 2

        # Settings for MNIST data
        _cfg.DATA.MNIST = CN()
        _cfg.DATA.MNIST.VAL_PERCENT = 0.2
        _cfg.DATA.MNIST.BINARY = [2, 3] # leave empty to have all classes

        # Override parameter values from YAML file first, then from override list.
        self._cfg = _cfg
        if config_file is not None:
            self._cfg.merge_from_file(str(config_file))
        self._cfg.merge_from_list(config_override)

        # Do any sort of validations required for the config.
        self._validate()

        # Make an instantiated object of this class immutable.
        self._cfg.freeze()

    
    def dump(self, file_path: Path):
        r"""
        Save config at the specified file path.
        Parameters
        ----------
        file_path: str
            (YAML) path to save config at.
        """
        file_path.parent.mkdir(exist_ok=True, parents=True)
        self._cfg.dump(stream=open(file_path, "w"))

    def _validate(self):
        r"""
        Perform all validations to raise error if there are parameters with conflicting values.
        """
        return True
        # if self._C.MODEL.USE_CBS:
        #     assert self._C.MODEL.EMBEDDING_SIZE == 300, "Word embeddings must be initialized with"
        #     " fixed GloVe Embeddings (300 dim) for performing CBS decoding during inference. "
        #     f"Found MODEL.EMBEDDING_SIZE as {self._C.MODEL.EMBEDDING_SIZE} instead."

        # assert (
        #     self._C.MODEL.MIN_CONSTRAINTS_TO_SATISFY <= self._C.DATA.CBS.MAX_GIVEN_CONSTRAINTS
        # ), "Satisfying more constraints than maximum specified is not possible."

    def __getattr__(self, attr: str):
        return self._cfg.__getattr__(attr)

    def __str__(self):
        #common_string: str = str(CN({"RANDOM_SEED": self._C.RANDOM_SEED})) + "\n"
        common_string: str = str(CN({"Settings": self._cfg})) + "\n"

        return common_string

    def __repr__(self):
        return self._cfg.__repr__()