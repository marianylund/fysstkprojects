import json
import pathlib

# Inspired by the code from Håkon Hukkelås github.com/hukkelas

def write_json(data:dict, filepath: pathlib.Path): 
    with filepath.open(mode='w') as f:
        json.dump(data, f) 

def load_data_as_dict(filename) -> dict:
    with open(filename) as json_file: 
        data = json.load(json_file)
    return data

def load_data_as_model(filename):
    # TODO: Not sure if it will be needed though
    return

def save_data(run_info: dict, model_info: dict):
    # TODO: merge the two dictionaries and save them
    return

def save_checkpoint(state_dict: dict,
                    filepath: pathlib.Path,
                    is_best: bool = False,
                    max_keep: int = 1):
    """
    Saves state_dict to filepath. Deletes old checkpoints as time passes.
    If is_best is toggled, saves a checkpoint to best.json
    """
    filepath.parent.mkdir(exist_ok=True, parents=True)
    list_path = filepath.parent.joinpath("latest_checkpoint")

    # Save the file itself
    write_json(state_dict, filepath)

    # Save it under the name best if it is
    if is_best:
        write_json(state_dict, filepath.parent.joinpath("best.json"))
    
    # Get all the previous checkpoints
    previous_checkpoints = get_previous_checkpoints(filepath.parent)

    # If the file is not in the list, add it
    if filepath.name not in previous_checkpoints:
        previous_checkpoints = [filepath.name] + previous_checkpoints

    # Delete the old ones 
    if len(previous_checkpoints) > max_keep:
        for ckpt in previous_checkpoints[max_keep:]:
            path = filepath.parent.joinpath(ckpt)
            if path.exists():
                path.unlink()

    previous_checkpoints = previous_checkpoints[:max_keep]
    with open(list_path, 'w') as fp:
        fp.write("\n".join(previous_checkpoints))

def get_previous_checkpoints(directory: pathlib.Path) -> list:
    assert directory.is_dir()
    list_path = directory.joinpath("latest_checkpoint")
    list_path.touch(exist_ok=True)

    with open(list_path) as fp:
        ckpt_list = fp.readlines()

    return [_.strip() for _ in ckpt_list]

def load_best_checkpoint(directory: pathlib.Path) -> dict:
    filepath = directory.joinpath("best.json")

    if not filepath.is_file():
        return None

    return load_data_as_dict(directory.joinpath("best.json"))

def add_more_info_to_dict(orig: dict, add:dict) -> dict:
    for key in add.keys():
        orig[key] = add[key]
    return orig