from pathlib import Path
from typing import Dict

import torch
import yaml

from .model import Transformer


def load_hparams(hparams: Path | str) -> Dict:
    """
    Load the hyperparameters from a yaml file.
    :param hparams: the path of the yaml file
    :return: the hyperparameters in a dictionary
    """
    with open(hparams, 'r') as f:
        hparams = yaml.safe_load(f)
    return hparams


def create_model(hparams: Dict, checkpoint: Path | str) -> Transformer:
    """
    Create the model from the hyperparameters and the checkpoint.
    :param hparams: the hyperparameters
    :param checkpoint: the path of the checkpoint
    :return: the model
    """
    model = Transformer(vocab_size=hparams["vocab_size"], layers=hparams["layers"],
                        n_heads=hparams["n_heads"], embeddings_size=hparams["embeddings_size"],
                        dropout=hparams["dropout"], bias=hparams["bias"],
                        context_size=hparams["context_size"])
    model.load_state_dict(torch.load(checkpoint))
    return model
