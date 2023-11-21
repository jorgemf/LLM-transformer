import enum
import os
import re
from typing import Optional, Tuple

import datasets
import numpy as np
import requests
import torch
import typer
from datasets import interleave_datasets, Dataset
from torch.utils.data import DataLoader, random_split
from transformers.data.data_collator import DataCollatorWithPadding
from transformers.tokenization_utils import PreTrainedTokenizer

from .config import TOKEN, LANGUAGES

app = typer.Typer()


class DatasetType(enum.Enum):
    """
    The type of dataset to use.
    """
    FULL = "full"
    SMALL = "small"


class TextDataset(Dataset):
    """
    A dataset that returns the text as a sequence of tokens.
    """

    def __init__(self, tokens: np.ndarray, seq_length: int = 1024):
        """
        Initializes the dataset.
        :param tokens: the data to use
        :param seq_length: the length of the sequence
        """
        self.tokens = tokens
        self.seq_length = seq_length

    def __len__(self):
        return len(self.tokens) - self.seq_length

    def __getitem__(self, idx):
        inputs_ids = [np.copy(self.tokens[x:x + self.seq_length]).astype(np.int64) for x in idx]
        attention_mask = [np.ones(len(inputs_ids[x]), dtype=np.int32) for x in range(len(idx))]
        return {
            "input_ids": inputs_ids,
            "attention_mask": attention_mask
        }


def get_dataloader_small(tokenizer: PreTrainedTokenizer, sequence_length: int, batch_size: int,
                         train: bool = True) \
        -> Tuple[DataLoader, DataLoader]:
    """
    Returns the dataloader for training and validation. The validation dataset is the initial part
    of the training dataset.

    :param tokenizer: the tokenizer to use
    :param sequence_length: the maximum length of the sequence
    :param batch_size: the batch size
    :param train: whether the dataset is for training or not (it only shuffles the data)
    :return: a tuple of (train_dataloader, val_dataloader)
    """
    file_path = "data/tinyshakespeare.txt"
    if not os.path.exists(file_path):
        print(f"Downloading tinyshakespeare dataset...")
        data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        with open(file_path, 'w') as f:
            f.write(requests.get(data_url).text)
    bin_file_path = f"data/tinyshakespeare_"
    if not os.path.exists(bin_file_path + "train.bin"):
        print("Processing dataset...")
        with open(file_path, "r") as f:
            data = f.read()
        split_point = int(len(data) * 0.9)
        train_data = data[:split_point]
        val_data = data[split_point:]
        for split, data in zip(["train", "val"], [train_data, val_data]):
            tokens = tokenizer(data)['input_ids']
            tokens_np = np.array(tokens, dtype=np.uint16)
            tokens_np.tofile(bin_file_path + split + ".bin")
    train_data = np.memmap(bin_file_path + "train.bin", dtype=np.uint16, mode='r')
    val_data = np.memmap(bin_file_path + "val.bin", dtype=np.uint16, mode='r')
    print(f"Train has {len(train_data)} tokens")
    print(f"Validation has {len(val_data)} tokens")
    train_dataloader = DataLoader(TextDataset(train_data, seq_length=sequence_length),
                                  batch_size=batch_size,
                                  shuffle=train,
                                  pin_memory=True,
                                  num_workers=os.cpu_count())
    val_dataloader = DataLoader(TextDataset(val_data, seq_length=sequence_length),
                                batch_size=batch_size,
                                shuffle=False,
                                drop_last=False,
                                pin_memory=False,
                                num_workers=os.cpu_count() // 2)

    return train_dataloader, val_dataloader


@app.command()
def download():
    """
    Downloads the dataset.
    """
    for lang in LANGUAGES:
        print(f"Downloading {lang} files...")
        datasets.load_dataset("bigcode/the-stack-dedup", data_dir=f"data/{lang}", split="train",
                              use_auth_token=TOKEN, num_proc=8)
    print("Done.")


def get_dataloader(tokenizer: PreTrainedTokenizer, sequence_length: int, batch_size: int,
                   max_code_generated: Optional[int] = None, finetune: bool = False,
                   train: bool = True, val_samples: Optional[int] = None,
                   chars_per_token: float = 3.4) \
        -> Tuple[DataLoader, DataLoader]:
    """
    Returns the dataloader for training and validation. The validation dataaset is the initial part
    of the training dataset.

    :param tokenizer: the tokenizer to use
    :param sequence_length: the maximum length of the sequence
    :param batch_size: the batch size
    :param max_code_generated: the maximum amount of code generated
    :param finetune: whether to finetune or not
    :param train: whether the dataset is for training or not (it only shuffles the data)
    :param val_samples: number of samples to use for validation
    :param chars_per_token: characters per token on average
    :return: a tuple of (train_dataloader, val_dataloader)
    """
    assert train or not finetune, "Finetune is only supported for train=True"
    all_ds = [datasets.load_dataset("bigcode/the-stack-dedup",
                                    data_dir=f"data/{lang}",
                                    split="train",
                                    num_proc=8,
                                    token=TOKEN) for lang in LANGUAGES]
    sizes = [len(ds) for ds in all_ds]
    sum_sizes = sum(sizes)
    probabilities = [size / sum_sizes for size in sizes]
    ds = interleave_datasets(all_ds, probabilities=probabilities,
                             stopping_strategy="all_exhausted")
    ds = ds.remove_columns([x for x in ds.column_names if x != "content"])

    def _basic_tokenizer(example):
        text_list = example["content"]
        for idx in range(len(text_list)):
            text = text_list[idx]
            sequence_chars = int(sequence_length * chars_per_token)
            if len(text) > sequence_chars:
                index = torch.randint(high=len(text) - sequence_chars, size=(1,))
                text_list[idx] = text[index:index + sequence_chars]
        return tokenizer(text_list, max_length=sequence_length, truncation=True,
                         return_token_type_ids=False, return_tensors='np')

    if finetune:
        ds.set_transform(CompleteCodeDataset(tokenizer=tokenizer,
                                             seq_length=sequence_length,
                                             max_code_generated=max_code_generated,
                                             chars_per_token=chars_per_token))
    else:
        ds.set_transform(_basic_tokenizer)

    data_collator = DataCollatorWithPadding(tokenizer, padding="longest",
                                            max_length=sequence_length, return_tensors='pt')

    if val_samples is not None:
        generator = torch.Generator().manual_seed(42)
        val_ds, ds = random_split(ds, [val_samples, len(ds) - val_samples], generator=generator)
        val_dataloader = DataLoader(val_ds,
                                    batch_size=batch_size,
                                    collate_fn=data_collator,
                                    shuffle=False,
                                    drop_last=False,
                                    pin_memory=False,
                                    num_workers=os.cpu_count() // 2)
    else:
        val_dataloader = None
    train_dataloader = DataLoader(ds,
                                  batch_size=batch_size,
                                  collate_fn=data_collator,
                                  shuffle=train,
                                  drop_last=False,
                                  pin_memory=True,
                                  num_workers=os.cpu_count())
    return train_dataloader, val_dataloader


class CompleteCodeDataset(object):
    """
    A special object to modify the code to use it for code generation. It takes a random part of
    the code that it is supposed to be generated (between two blank spaces) and based on the context
    (the code before of what we are going to generate and the code after of what we are going to
    generate) it generates the code. So the sequence is:
    previous context + separator (sep) + next context + separator (bos) + code to generate + (eos)
    """

    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            seq_length: int = 1024,
            max_code_generated: int = 256,
            input_column_name: str = "content",
            chars_per_token: float = 3.5,
    ):
        """

        :param tokenizer: the tokenizer to use
        :param seq_length: the length of the sequence
        :param max_code_generated: the maximum amount of code generated
        :param input_column_name: the input column name
        :param chars_per_token: the amount of characters per token on average
        """
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.seq_length_chars = int(seq_length * chars_per_token)
        self.max_code_generated_chars = int(max_code_generated * chars_per_token)
        self.input_column_name = input_column_name
        self.white_spaces_re = re.compile(r'[\s\n\t]')
        self.new_line_re = re.compile(r'[\n]')

    def __call__(self, data):
        text_list = data[self.input_column_name]
        for idx in range(len(text_list)):
            text = text_list[idx]
            initial_pos = torch.randint(0, len(text), (1,)).item()
            match = self.new_line_re.search(text, initial_pos)
            if match is not None:
                initial_pos = match.start() + 1
                prediction_length = torch.randint(0, self.max_code_generated_chars, (1,)).item()
                final_pos = self.new_line_re.search(text, initial_pos + prediction_length)
                if final_pos is not None:
                    final_pos = final_pos.start()
                    context_window = self.seq_length_chars - (final_pos + initial_pos) - 2
                    pre_code = text[max(0, initial_pos - context_window):initial_pos]
                    post_code = text[final_pos:final_pos + context_window]
                    target = text[initial_pos:final_pos]
                    text_list[idx] = (pre_code + self.tokenizer.sep_token + post_code +
                                      self.tokenizer.bos_token + target + self.tokenizer.eos_token)
        return self.tokenizer(text=text_list, max_length=self.seq_length, truncation=True,
                              return_token_type_ids=False, return_special_tokens_mask=False)
