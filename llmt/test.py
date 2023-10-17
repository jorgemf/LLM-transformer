from typing import Dict

import numpy as np
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .model import Transformer
from .tokenizer import ReplitLMTokenizer


def test(val_samples: int = 10000):
    # TODO: implement the test function
    raise NotImplementedError


def test_epoch(accelerator: Accelerator,
               model: Transformer,
               tokenizer: ReplitLMTokenizer,
               val_dataloader: DataLoader,
               context_size: int,
               finetune: bool,
               device: torch.device,
               epoch: int,
               global_step: int,
               profiler=None) -> Dict[str, float]:
    """
    Test the model for one epoch.
    :param accelerator: the accelerator to use
    :param model: the model to use
    :param tokenizer: the tokenizer to use
    :param val_dataloader: the validation dataloader to use
    :param context_size: the context size to use
    :param finetune: whether to finetune the model or not
    :param device: the device to use
    :param epoch: current epoch
    :param global_step: current global step
    :param profiler: the profiler to use
    :return: a dictionary with the metrics, the loss included
    """
    losses = []
    separator_token = tokenizer.bos_token_id if finetune else None
    with torch.no_grad():
        model.eval()
        generated_text = ""
        generated_main = ""
        if accelerator.is_main_process:
            model_unwrap = accelerator.unwrap_model(model)
            generated_text = generate_text_sequence(model_unwrap, context_size, tokenizer, device)
            generated_main = generate_text_sequence(model_unwrap, context_size, tokenizer, device,
                                                    initial_text="fun main(args: Array<String>) {")

        total_steps = len(val_dataloader.dataset)
        progress_bar = tqdm(total=total_steps, leave=False, position=0,
                            disable=not accelerator.is_local_main_process)
        for batch, data in enumerate(val_dataloader):
            progress_bar.set_description(f"Step {batch + 1}/{total_steps}")
            tokens = data["input_ids"].to(dtype=torch.long)
            mask = data["attention_mask"].to(dtype=torch.long)[:, :-1]
            sequence = tokens[:, :-1]
            target = tokens[:, 1:]
            _, loss = model(sequence, targets=target, mask=mask, sep_token=separator_token)

            gathered_loss = accelerator.gather_for_metrics(loss.item())
            if type(gathered_loss) is list:
                losses.extend(gathered_loss)
            else:
                losses.append(gathered_loss)
            logs = {"loss": np.mean(losses)}
            progress_bar.set_postfix(**logs)
            progress_bar.update(np.sum(accelerator.gather_for_metrics(tokens.shape[0])))
            if profiler is not None:
                profiler.step()
    loss = np.mean(losses)
    return {
        'loss': loss,
        'perplexity': np.exp(loss),
        'generated_text:': generated_text,
        'generated_main:': generated_main
    }


def generate_text_sequence(model: Transformer, context_size: int, tokenizer: ReplitLMTokenizer,
                           device: torch.device, initial_text: str = "") -> str:
    """
    Generate a text sequence from the model.
    :param model: the model to use
    :param context_size: the context size to use
    :param tokenizer: the tokenizer to use
    :param device: the device to use
    :param initial_text: the initial text to use, if empy if generates a sequence from scratch
    :return: a text sequence
    """
    if len(initial_text) == 0:
        sequences = torch.zeros((1, 1), dtype=torch.long, device=device)
    else:
        tokens = tokenizer(initial_text)['input_ids']
        sequences = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    sequences = model.generate(sequences, context_size, top_k=1000)
    generated_text = tokenizer.decode(sequences[0].cpu().numpy())
    return generated_text
