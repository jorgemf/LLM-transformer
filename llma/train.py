from datetime import datetime
from typing import Optional

from tqdm.auto import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import profile, ProfilerActivity
from torch.profiler.profiler import tensorboard_trace_handler
from torch.optim.lr_scheduler import ExponentialLR
from .config import get_device, RUNS_DIR, DTYPE
from .dataset import get_dataloader
from .model import Transformer
from .test import test_epoch
from .tokenizer import ReplitLMTokenizer


def train(batch_size: int = 22,
          epochs: int = 3,
          learning_rate: float = 1e-4,
          learning_rate_decay: float = 0.1,
          layers: int = 14,
          heads: int = 8,
          embed_dim: int = 2 * 1024,
          context_size: int = 512,
          dropout: float = 0.1,
          bias: bool = True,
          finetune: bool = True,
          profiler: bool = False,
          experiment_dir=None,
          val_samples: Optional[int] = 200) -> None:
    """
    Trains the model.
    :param batch_size: size of the batch
    :param epochs: number of epochs to train the model
    :param learning_rate: initial learning rate
    :param learning_rate_decay: learning rate decay
    :param layers: number of layers of the model
    :param heads: number of attention heads of the model
    :param embed_dim: embbeding dimension for the attention (all heads combined)
    :param context_size: context size of the model to make predictions
    :param dropout: whether to use dropout or not (value between 0 and 1 for the dropout rate)
    :param bias: whether to use bias or not
    :param finetune: whether to finetune the model or not (data is adapted for code prediction)
    :param profiler: whether to profile the code or not
    :param experiment_dir: directory to save the experiment
    :param val_samples: number of samples to use for validation (it uses the initial samples of the
     data)
    :return:
    """
    dtype = DTYPE
    if experiment_dir is None:
        experiment_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_dir = RUNS_DIR.joinpath(experiment_dir)
    writer = SummaryWriter(experiment_dir.joinpath("logs/train"))
    tokenizer = ReplitLMTokenizer(truncation_side="left" if finetune else "right")

    device = get_device()
    model = Transformer(vocab_size=tokenizer.vocab_size,
                        layers=layers,
                        n_heads=heads,
                        embeddings_size=embed_dim,
                        dropout=dropout,
                        bias=bias,
                        context_size=context_size,
                        ).to(device).to(dtype)
    print("Model parameters:", model.get_num_params())

    betas = (0.9, 0.95)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=betas,
                                  weight_decay=1e-1)
    lr_scheduler = ExponentialLR(optimizer=optimizer, gamma=learning_rate_decay)

    train_dataloader, val_dataloader = get_dataloader(tokenizer=tokenizer,
                                                      sequence_length=context_size,
                                                      batch_size=batch_size,
                                                      train=True,
                                                      val_samples=val_samples)

    prof = None
    if profiler:
        def _trace_handler(prof):
            print("Profiling...")
            tensorboard_trace_handler(dir_name=experiment_dir, use_gzip=True)(prof)
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

    progress_bar = tqdm(total=epochs, leave=True, position=0)
    progress_bar.set_description(f"Epoch")
    if profiler:
        prof.start()
    size_dataset = len(train_dataloader.dataset)

    best_score = test_epoch(model, tokenizer, val_dataloader, context_size, finetune, writer,
                            device=device, epoch=0, global_step=0)

    for e in range(epochs):
        progress_bar.set_description(f"Epoch {e + 1}/{epochs}")
        epoch(train_dataloader, tokenizer, e, model, optimizer, lr_scheduler, finetune, writer,
              device=device, dtype=dtype)
        progress_bar.update(1)
        lr_scheduler.step()
        score = test_epoch(model, tokenizer, val_dataloader, context_size, finetune, writer,
                           device=device, epoch=e, global_step=(e + 1) * size_dataset)
        if (best_score is None) or (score < best_score):
            best_score = score
            print(f"New best score: {best_score}")
            torch.save({
                'global_step': (e + 1) * size_dataset,
                'model': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, experiment_dir.joinpath(f"model_{e + 1}_{best_score:.4f}.pt"))
    if profiler:
        prof.stop()
    writer.close()


def epoch(dataloader: torch.utils.data.DataLoader,
          tokenizer: ReplitLMTokenizer,
          current_epoch: int,
          model: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
          finetune: bool,
          summary_writer: SummaryWriter,
          device: torch.device,
          dtype: torch.dtype,
          profiler: Optional[profile] = None) -> None:
    """
    Trains the model for one epoch.
    :param dataloader: the data
    :param tokenizer: the tokenizer
    :param current_epoch: current epoch
    :param model: the model
    :param optimizer: the optimizer
    :param lr_scheduler: the learning rate scheduler
    :param finetune: whether to finetune the model or not (data is adapted for code prediction)
    :param summary_writer: the summary writer to log the metrics
    :param device: the device to use
    :param dtype: the data type to use
    :param profiler: the profiler to use
    :return: None
    """
    model.train()

    separator_token = tokenizer.bos_token_id if finetune else None
    total_steps = len(dataloader)
    total_examples = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    progress_bar = tqdm(total=total_steps, leave=False, position=1)
    progress_bar.set_description(f"Step")
    for batch, data in enumerate(dataloader):

        # For using a bigger batch size that the GPU supports:
        # https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation
        # https://huggingface.co/docs/diffusers/main/en/tutorials/basic_training

        tokens = data["input_ids"].to(device=device, dtype=torch.long)
        sequence = tokens[:, :-1]
        target = tokens[:, 1:]
        mask = data["attention_mask"].to(device=device, dtype=torch.bool)[:, :-1]
        _, loss = model(sequence, targets=target, mask=mask, sep_token=separator_token)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Logging
        step = tokens.shape[0] + batch * batch_size + total_examples * current_epoch
        summary_writer.add_scalar('loss', loss.item(), global_step=step)

        # update progress bar
        progress_bar.update(1)
        logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
        progress_bar.set_postfix(**logs)

        if profiler is not None:
            profiler.step()
