from pathlib import Path
from typing import Dict, Tuple

import torch
import yaml

from .model import Transformer
from .tokenizer import ReplitLMTokenizer
from .config import get_device


def load_hparams(hparams: Path | str) -> Dict:
    """
    Load the hyperparameters from a yaml file.
    :param hparams: the path of the yaml file
    :return: the hyperparameters in a dictionary
    """
    with open(hparams, 'r') as f:
        hparams = yaml.safe_load(f)
    return hparams


def create_model(hparams: Dict, checkpoint: Path | str) -> Tuple[Transformer, ReplitLMTokenizer]:
    """
    Create the model from the hyperparameters and the checkpoint.
    :param hparams: the hyperparameters
    :param checkpoint: the path of the checkpoint
    :return: the model
    """
    tokenizer = ReplitLMTokenizer()
    device = get_device()
    model = Transformer(vocab_size=tokenizer.vocab_size, layers=hparams["layers"],
                        n_heads=hparams["heads"], embeddings_size=hparams["embed_dim"],
                        dropout=hparams["dropout"], bias=hparams["bias"],
                        context_size=hparams["context_size"],
                        focus_percent=hparams["focus_percent"], focus_temp=hparams["focus_temp"],
                        focus_min_seq_len=hparams["focus_min_seq_len"],
                        focus_num_fixed_positions=hparams["focus_num_fixed_positions"])
    model.load_state_dict(torch.load(checkpoint))
    model.to(device)
    return model, tokenizer
