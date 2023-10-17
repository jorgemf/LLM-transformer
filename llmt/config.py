from pathlib import Path

import datasets
import torch

# programming languages to use for training the model
# LANGUAGES = ["kotlin", "python"]
LANGUAGES = ["kotlin"]
# the base directory to store the experiments or runs
RUNS_DIR = Path("./runs")
# the data type to use for the model
DTYPE = torch.bfloat16
# the authentication token to download the dataset (it is downloaded in the setup function)
TOKEN = ""
# the time interval to save the checkpoint
CHECKPOINT_SAVE_TIME = 30 * 60  # 30 minutes


def setup():
    """
    Sets up the config:
    - sets the downloaded datasets path
    - sets the cache path
    - sets the authentication token to download the dataset
    - sets the device and checks if the gpu is available
    - creates the runs directory
    """
    global TOKEN
    print("Setting up config...")
    datasets.config.DOWNLOADED_DATASETS_PATH = Path("./data/datasets")
    datasets.config.HF_DATASETS_CACHE = Path("./data/cache")

    with open("data/token", "r") as token_file:
        TOKEN = token_file.read().strip()

    device = get_device()
    print(f"Using {device} device")

    RUNS_DIR.mkdir(exist_ok=True)


def get_device() -> str:
    """
    :return: the device to use for training and testing
    """
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    return device


def get_mixed_precission(dtype) -> str:
    """
    Gets the mixed precision type.
    :param dtype: the data type
    :return:
    """
    if dtype == torch.float16:
        return 'bf16'
    elif dtype == torch.bfloat16:
        return 'fp16'
    else:
        return 'no'
