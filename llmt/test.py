from typing import Optional

import torch
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from accelerate import Accelerator
import numpy as np
from .tokenizer import ReplitLMTokenizer
from .model import Transformer


def test(val_samples: int = 10000):
    # TODO: implement the test function
    raise NotImplementedError


def test_epoch(accelerator: Accelerator,
               model: Transformer,
               tokenizer: ReplitLMTokenizer,
               val_dataloader: DataLoader,
               context_size: int,
               finetune: bool,
               summary_writer: Optional[SummaryWriter],
               device: torch.device,
               epoch: int,
               global_step: int,
               profiler=None,
               tqdm_position: int = 0) -> float:
    total_steps = len(val_dataloader.dataset)
    progress_bar = tqdm(total=total_steps, leave=False, position=tqdm_position,
                        disable=not accelerator.is_local_main_process)
    losses = []
    separator_token = tokenizer.bos_token_id if finetune else None
    with torch.no_grad():
        model.eval()
        if accelerator.is_main_process and summary_writer is not None:
            model_unwrap = accelerator.unwrap_model(model)
            generated_text = generate_text_sequence(model_unwrap, context_size, tokenizer, device)
            print("generated text:", generated_text)
            summary_writer.add_text("test/generated", generated_text, global_step=global_step)
            generated_text = generate_text_sequence(model_unwrap, context_size, tokenizer, device,
                                                    initial_text="fun main(args: Array<String>) {")
            print("generated main:", generated_text)
            summary_writer.add_text("test/generated_main", generated_text, global_step=global_step)

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

    if accelerator.is_main_process and summary_writer is not None:
        summary_writer.add_scalar("test/loss", loss, global_step=global_step)
    return loss


def generate_text_sequence(model: Transformer, context_size:int, tokenizer: ReplitLMTokenizer,
                           device: torch.device, initial_text: str = "") -> str:
    """
    Generate a text sequence from the model.
    :param model: the model to use
    :param context_size: the context size to use
    :param tokenizer: the tokenizer to use
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