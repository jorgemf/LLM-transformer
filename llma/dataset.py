import os
from typing import Optional, Tuple
import re
import typer
import torch
import datasets
from torch.utils.data import DataLoader
from datasets import load_dataset, interleave_datasets, Dataset
from transformers.data.data_collator import DataCollatorWithPadding
from transformers.tokenization_utils import PreTrainedTokenizer

from .config import TOKEN, LANGUAGES

app = typer.Typer()


def get_small_dataset() -> Dataset:
    """
    :return: a small dataset for testing so you don't have to wait to load the full data
    """
    if os.path.exists("data/llm_val_text.hf"):
        print("Loading cached val text...")
        ds = datasets.load_from_disk("data/llm_val_text.hf")
    else:
        all_ds = [datasets.load_dataset("bigcode/the-stack-dedup",
                                        data_dir=f"data/{lang}",
                                        split="train",
                                        use_auth_token=TOKEN) for lang in
                  LANGUAGES]
        ds = interleave_datasets(all_ds, stopping_strategy="all_exhausted")
        if filter is not None:
            ds = ds.select(range(100000))

        data = list(ds)
        columns = {key: [item[key] for item in data] for key in data[0].keys()}
        hf_dataset = Dataset.from_dict(columns)
        hf_dataset.save_to_disk("data/llm_val_text.hf")
    return ds


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


@app.command()
def process():
    """
    Processes the dataset to make the load speed faster. (Not implemented yet)
    """
    raise NotImplementedError()  # TODO


def get_dataloader(tokenizer: PreTrainedTokenizer, sequence_length: int, batch_size: int,
                   max_code_generated: Optional[int] = None, finetune: bool = False,
                   train: bool = True, val_samples: Optional[int] = None,
                   chars_per_token: float = 3.5) \
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
    :param chars_per_token:
    :return: a tuple of (train_dataloader, val_dataloader)
    """
    assert train or not finetune, "Finetune is only supported for train=True"
    ds = get_small_dataset()
    # all_ds = [datasets.load_dataset("bigcode/the-stack-dedup",
    #                                 data_dir=f"data/{lang}",
    #                                 split="train",
    #                                 num_proc=8,
    #                                 use_auth_token=TOKEN) for lang in
    #           LANGUAGES]
    # sizes = [len(ds) for ds in all_ds]
    # sum_sizes = sum(sizes)
    # probabilities = [size / sum_sizes for size in sizes]
    # ds = interleave_datasets(all_ds, probabilities=probabilities,
    #                          stopping_strategy="all_exhausted")
    ds = ds.remove_columns([x for x in ds.column_names if x != "content"])

    def _basic_tokenizer(example):
        text = example["content"]
        if len(text) > sequence_length * chars_per_token:
            index = torch.randint(high=len(example) - sequence_length * chars_per_token, size=(1,))
            text = text[index:index + sequence_length * chars_per_token]
        return tokenizer(text, return_token_type_ids=False, truncation=True,
                         max_length=sequence_length, return_tensors='np')

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
        val_ds = ds.select(range(val_samples))
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


class CompleteCodeDataset(Dataset):
    """
    A special dataset to modify the code to use it for code generation. It takes a random part of
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
        self.seq_length = int(seq_length * chars_per_token)
        self.max_code_generated = int(max_code_generated * chars_per_token)
        self.input_column_name = input_column_name
        self.white_spaces_re = re.compile(r'[\s\n\t]')
        self.new_line_re = re.compile(r'[\n]')

    def __call__(self, data):
        text = data[self.input_column_name]
        initial_pos = torch.randint(0, len(text), (1,)).item()
        match = self.new_line_re.search(text, initial_pos)
        if match is not None:
            initial_pos = match.start() + 1
            prediction_length = torch.randint(0, self.max_code_generated, (1,)).item()
            final_pos = self.new_line_re.search(text, initial_pos + prediction_length)
            if final_pos is not None:
                final_pos = final_pos.start()
                context_window = self.seq_length - (final_pos + initial_pos) - 2
                pre_code = text[max(0, initial_pos - context_window):initial_pos]
                post_code = text[final_pos:final_pos + context_window]
                target = text[initial_pos:final_pos]
                text = (pre_code + self.tokenizer.sep_token + post_code + self.tokenizer.bos_token +
                        target + self.tokenizer.eos_token)
        return self.tokenizer(text=text, return_token_type_ids=False,
                              truncation_side="left", max_length=self.seq_length,
                              truncation=True,
                              return_special_tokens_mask=False)
