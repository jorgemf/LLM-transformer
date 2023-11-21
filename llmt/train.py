import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Tuple

import numpy as np
import torch
import yaml
from accelerate import Accelerator
from torch.optim.lr_scheduler import ExponentialLR
from torch.profiler import profile, ProfilerActivity
from torch.profiler.profiler import tensorboard_trace_handler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams
from tqdm.auto import tqdm

from .config import RUNS_DIR, DTYPE, get_device, CHECKPOINT_SAVE_TIME
from .dataset import DatasetType, get_dataloader, get_dataloader_small
from .model import Transformer
from .test import test_epoch
from .tokenizer import ReplitLMTokenizer

checkpoint_time = time.time() + CHECKPOINT_SAVE_TIME


def train(batch_size: int = 28,
          gradient_accumulation_steps: int = 10,
          epochs: int = 5,
          learning_rate: float = 1e-3,
          learning_rate_decay: float = 0.5,
          layers: int = 12,
          heads: int = 12,
          embed_dim: int = 768,
          context_size: int = 1024,
          max_code_generated: int = 256,
          bias: bool = True,
          dropout: float = 0.0,
          gradient_clip: float = 1.0,
          weight_decay: float = 1e-1,
          focus_temp: float = 0.8,
          focus_percent: float = 0.85,
          focus_min_seq_len: int = 10,
          focus_num_fixed_positions: int = 30,
          finetune: bool = False,
          profiler: bool = False,
          experiment_dir: str = None,
          val_samples: Optional[int] = 200,
          dataset_type: DatasetType = DatasetType.FULL.value,
          checkpoint: str = None,
          initial_val: bool = False) -> None:
    """
    Trains the model.
    :param batch_size: size of the mini-batch
    :param gradient_accumulation_steps: number of steps to accumulate the gradients before
    backpropagation
    :param epochs: number of epochs to train the model
    :param learning_rate: initial learning rate
    :param learning_rate_decay: learning rate decay
    :param layers: number of layers of the model
    :param heads: number of attention heads of the model
    :param embed_dim: embbeding dimension for the attention (all heads combined)
    :param context_size: context size of the model to make predictions
    :param max_code_generated: maximum amount of code generated
    :param bias: whether to use bias or not
    :param dropout: whether to use dropout or not (value between 0 and 1 for the dropout rate)
    :param gradient_clip: gradient clipping, use 0.0 to disable it
    :param weight_decay: weight decay for the optimizer (regularization)
    :param focus_temp: temperature for the focus mechanism
    :param focus_percent: percentage of the sequence to focus on
    :param focus_min_seq_len: minimum sequence length to apply the focus attention
    :param focus_num_fixed_positions: number of fixed positions to use for the focus attention
    :param finetune: whether to finetune the model or not (data is adapted for code prediction)
    :param profiler: whether to profile the code or not
    :param experiment_dir: directory to save the experiment
    :param val_samples: number of samples to use for validation (it uses the initial samples of the
     data)
    :param dataset_type: the type of dataset to use, full for code and small for testing with text
    :param checkpoint: path to a checkpoint to continue training
    :param initial_val: whether to perform the initial validation or not
    :return:
    """
    global checkpoint_time
    assert dataset_type == DatasetType.FULL or not finetune, \
        "finetune only is available with full dataset"
    dtype = DTYPE
    if experiment_dir is None:
        experiment_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_dir = RUNS_DIR.joinpath(experiment_dir)
    config = locals()
    config['experiment_dir'] = str(experiment_dir)
    config['dtype'] = str(dtype)
    config['dataset_type'] = dataset_type.name
    device = get_device()

    tokenizer = ReplitLMTokenizer(truncation_side="left" if finetune else "right",
                                  model_max_length=context_size)

    model = Transformer(vocab_size=tokenizer.vocab_size,
                        layers=layers,
                        n_heads=heads,
                        embeddings_size=embed_dim,
                        dropout=dropout,
                        bias=bias,
                        context_size=context_size,
                        focus_temp=focus_temp,
                        focus_percent=focus_percent,
                        focus_min_seq_len=focus_min_seq_len,
                        focus_num_fixed_positions=focus_num_fixed_positions,
                        ).to(dtype)

    betas = (0.9, 0.95)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=betas,
                                  weight_decay=weight_decay)
    lr_scheduler = ExponentialLR(optimizer=optimizer, gamma=learning_rate_decay)

    if dataset_type == DatasetType.SMALL:
        train_dataloader, val_dataloader = get_dataloader_small(tokenizer=tokenizer,
                                                                sequence_length=context_size,
                                                                batch_size=batch_size,
                                                                train=True)
    elif dataset_type == DatasetType.FULL:
        train_dataloader, val_dataloader = get_dataloader(tokenizer=tokenizer,
                                                          sequence_length=context_size,
                                                          batch_size=batch_size,
                                                          finetune=finetune,
                                                          max_code_generated=max_code_generated,
                                                          train=True,
                                                          val_samples=val_samples)
    else:
        raise ValueError(f"Invalid dataset type: {dataset_type}")

    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)
    accelerator.print("Model parameters:", model.get_num_params())
    accelerator.print("Tokenizer dictionary:", tokenizer.vocab_size)

    model, optimizer, lr_scheduler, train_dataloader, val_dataloader = \
        accelerator.prepare(model, optimizer, lr_scheduler, train_dataloader, val_dataloader)

    if checkpoint is not None:
        accelerator.load_state(checkpoint)

    # logger
    writer_train, writer_val = None, None
    if accelerator.is_main_process:
        writer_train, writer_val = configure_logger(config, experiment_dir)
        for s in hparams(config, metric_dict={'perplexity': 0, 'loss': 0, 'lr': 0}):
            writer_val.file_writer.add_summary(s)
        writer_val.flush()

    # profiler
    prof = None
    if profiler and accelerator.is_main_process:
        prof = configure_profiler(experiment_dir)
    if prof is not None:
        prof.start()

    # progress bar
    progress_bar = tqdm(total=epochs, leave=True, position=0,
                        disable=not accelerator.is_local_main_process)
    progress_bar.set_description(f"Epoch")
    size_dataset = len(train_dataloader.dataset)

    # initial metrics
    best_score = None
    if initial_val:
        metrics = test_epoch(accelerator, model, tokenizer, val_dataloader, context_size, finetune,
                             device=device, epoch=0, global_step=0)
        best_score = metrics['loss']
        if accelerator.is_main_process:
            log_metrics(metrics, global_step=0, writer=writer_val)

    checkpoint_time = time.time() + CHECKPOINT_SAVE_TIME

    # training
    for e in range(epochs):
        progress_bar.set_description(f"Epoch {e + 1}/{epochs}")

        epoch(accelerator, train_dataloader, tokenizer, e, model, optimizer, lr_scheduler,
              finetune, gradient_clip, writer_train, dtype=dtype,
              batch_size=accelerator.num_processes * batch_size,
              save_path=experiment_dir.joinpath("checkpoint"))

        progress_bar.update(1)
        if accelerator.is_main_process:
            lr_scheduler.step()
        global_step = (e + 1) * size_dataset
        metrics = test_epoch(accelerator, model, tokenizer, val_dataloader, context_size, finetune,
                             device=device, epoch=e, global_step=global_step)

        if accelerator.is_main_process:
            score = metrics['loss']
            log_metrics(metrics, global_step=global_step, writer=writer_val)
            if (best_score is None) or (score < best_score):
                best_score = score
                accelerator.print(f"New best score: {best_score}")
                save_path = experiment_dir.joinpath(f"model_{e + 1}_{best_score:.4f}")
                accelerator.save_state(save_path)
                checkpoint_time = time.time() + CHECKPOINT_SAVE_TIME
    if accelerator.is_main_process:
        if prof is not None:
            prof.stop()
        writer_train.close()
        writer_val.close()


def log_metrics(metrics: Dict, global_step: int, writer: SummaryWriter):
    """
    Logs the metrics.
    :param conf: configuration of the experiment. if it is None the hdparams are not logged
    :param metrics: metrics to log
    :param global_step: current global step
    :param writer: summary writer
    """
    for key, value in metrics.items():
        if isinstance(value, (int, float, complex)):
            writer.add_scalar(key, value, global_step=global_step)
        elif isinstance(value, str):
            writer.add_text(key, value, global_step=global_step)
        else:
            raise ValueError(f"Invalid type for metric {key}: {type(value)}")
    writer.flush()


def configure_logger(config: Dict, experiment_dir: Path) -> Tuple[SummaryWriter, SummaryWriter]:
    """
    Configures the logger.
    :param config: the configuration of the experiment
    :param experiment_dir: the experiment directory
    :return: a tuple with the train and validation summary writers
    """
    writer_train = SummaryWriter(experiment_dir.joinpath("logs/train"), flush_secs=30)
    writer_val = SummaryWriter(experiment_dir.joinpath("logs/val"), flush_secs=30)
    # write local configuration
    with open(os.path.join(experiment_dir, "hparams.yml"), "w") as outfile:
        try:
            yaml.dump(config, outfile)
        except yaml.representer.RepresenterError:
            raise
    return writer_train, writer_val


def configure_profiler(experiment_dir: Path) -> profile:
    """
    Configures the profiler.
    :param experiment_dir: the experiment directory
    :return: the profiler
    """

    def _trace_handler(prof):
        print("Profiling...")
        tensorboard_trace_handler(dir_name=str(experiment_dir), use_gzip=True)(prof)
        for order in ["self_cpu_time_total", "cpu_time_total",
                      "self_cuda_time_total", "cuda_time_total"]:
            print(order)
            output = prof.key_averages().table(sort_by=order, row_limit=15)
            print(output)

    prof = profile(
        activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
        schedule=torch.profiler.schedule(skip_first=0, wait=2, warmup=2, active=2, repeat=1),
        on_trace_ready=_trace_handler,
        record_shapes=True,
        profile_memory=True,
        with_stack=True)
    print("Profiler ON", torch.profiler.supported_activities())
    return prof


def epoch(accelerator: Accelerator,
          dataloader: torch.utils.data.DataLoader,
          tokenizer: ReplitLMTokenizer,
          current_epoch: int,
          model: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
          finetune: bool,
          grad_clip: float,
          summary_writer: Optional[SummaryWriter],
          dtype: torch.dtype,
          batch_size: int,
          save_path: Path,
          profiler: Optional[profile] = None) -> None:
    """
    Trains the model for one epoch.
    :param accelerator: the accelerator to use
    :param dataloader: the data
    :param tokenizer: the tokenizer
    :param current_epoch: current epoch
    :param model: the model
    :param optimizer: the optimizer
    :param lr_scheduler: the learning rate scheduler
    :param finetune: whether to finetune the model or not (data is adapted for code prediction)
    :param grad_clip: gradient clipping, use 0.0 to disable it
    :param summary_writer: the summary writer to log the metrics
    :param dtype: the data type to use
    :param batch_size: the batch size (for all processes)
    :param save_path: the path to save the checkpoints
    :param profiler: the profiler to use
    :return: None
    """
    global checkpoint_time
    model.train()

    separator_token = tokenizer.bos_token_id if finetune else None
    total_steps = len(dataloader)
    total_examples = len(dataloader.dataset)

    progress_bar = tqdm(total=total_steps, leave=False, position=0,
                        disable=not accelerator.is_local_main_process)
    progress_bar.set_description(f"Epoch {current_epoch + 1}")

    for batch, data in enumerate(dataloader):

        with accelerator.accumulate(model):

            tokens = data["input_ids"].to(dtype=torch.long)
            sequence = tokens[:, :-1]
            target = tokens[:, 1:]
            mask = data["attention_mask"].to(dtype=torch.bool)[:, :-1]
            _, loss = model(sequence, targets=target, mask=mask, sep_token=separator_token)

            # Backpropagation (accumulated gradients with Accelerator)
            accelerator.backward(loss)
            # gradient clipping
            if grad_clip > 0.0 and accelerator.sync_gradients:
                accelerator.unscale_gradients(optimizer)
                accelerator.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()

            # Logging
            if accelerator.is_main_process and accelerator.sync_gradients and summary_writer is not None:
                current_batch_size = np.sum(accelerator.gather_for_metrics(tokens.shape[0]))
                step = current_batch_size + batch * batch_size + total_examples * current_epoch
                loss = np.mean(accelerator.gather_for_metrics(loss.item()))
                summary_writer.add_scalar('loss', loss, global_step=step)
                summary_writer.add_scalar('lr', lr_scheduler.get_last_lr()[0], global_step=step)
                logs = {"loss": loss, "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)

            # update progress bar and profiler step
            progress_bar.update(accelerator.num_processes)
            if profiler is not None:
                profiler.step()

            # save checkpoint just in case
            if time.time() > checkpoint_time:
                accelerator.save_state(save_path)
                checkpoint_time = time.time() + CHECKPOINT_SAVE_TIME
