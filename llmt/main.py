from typing import Optional

import torch
import typer

from .config import setup, get_device

setup()
from .dataset import app as dataset_app
from .train import train
from .test import test
from .utils import create_model, load_hparams

app = typer.Typer()
app.add_typer(dataset_app, name="dataset")
app.command()(train)
app.command()(test)


@app.command()
def generate(checkpoint: str, hparams: str, text: Optional[str] = None, topk: Optional[int] = None,
             temperature: Optional[float] = None, max_new_tokens: Optional[int] = None) -> None:
    """
    Generate text from a checkpoint.
    :param checkpoint: the path of the checkpoint to use
    :param hparams: the path of the hparams to use
    :param text: the initial text
    :param topk: the number of top tokens to consider for sampling
    :param temperature: the temperature to use for the softmax
    :param max_new_tokens: the maximum number of tokens to generate
    """
    hparams = load_hparams(hparams)
    model, tokenizer = create_model(hparams, checkpoint)
    model.eval()
    if text is None:
        text = ""
    if temperature is None:
        temperature = 1.0
    if max_new_tokens is None:
        max_new_tokens = model.context_size - len(text)

    device = get_device()
    if len(text) == 0:
        tokens = torch.zeros((1, 1), dtype=torch.long, device=device)
    else:
        tokens = tokenizer([text])['input_ids']
        tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    generated_tokens = model.generate(tokens, top_k=topk, temperature=temperature,
                                      max_new_tokens=max_new_tokens)
    generated_text = tokenizer.decode(generated_tokens[0].cpu().numpy())
    print(generated_text)


if __name__ == '__main__':
    app()
