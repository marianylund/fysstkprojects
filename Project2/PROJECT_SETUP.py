from pathlib import Path

ROJECT_ROOT_DIR = Path.cwd().joinpath("Results")
CHECKPOINTS_DIR = ROJECT_ROOT_DIR.joinpath("Checkpoints")
FIGURE_DIR = ROJECT_ROOT_DIR.joinpath("FigureFiles")
DATA_DIR = Path.cwd().joinpath("DataFiles")
TERRAIN_PATH = DATA_DIR.joinpath("SRTM_data_Norway_2.tif")

SEED = 2018
SAVE_FIG = False
