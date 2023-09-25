# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Forked from https://huggingface.co/replit/replit-code-v1-3b/blob/main/replit_lm_tokenizer.py
Forked from the file src/transformers/models/bert_generation/tokenization_bert_generation.py
from the HuggingFace Transformers library.
Permalink: https://github.com/huggingface/transformers/blob/04ab5605fbb4ef207b10bf2772d88c53fc242e83/src/transformers/models/bert_generation/tokenization_bert_generation.py

Tokenizer class for ReplitLM
Class is modified for compatibility with custom vocabulary and to achieve desired encode/decode
behavior for Replit Code V1 3B model.
"""
import sentencepiece as spm
from transformers import PreTrainedTokenizer
from typing import Any, Dict, List, Optional


class ReplitLMTokenizer(PreTrainedTokenizer):
    """
      Construct a ReplitLMTokenizer tokenizer. Based on [SentencePiece](https://github.com/google/sentencepiece).
      This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods.

      Args:
          vocab_file (`str`):
              [SentencePiece](https://github.com/google/sentencepiece) file (generally has a *.spm* extension) that
              contains the vocabulary necessary to instantiate a tokenizer.
          eos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
              The end of sequence token.
          bos_token (`str`, *optional*, defaults to `None`):
              The begin of sequence token.
          unk_token (`str`, *optional*, defaults to `"<|unk|>"`):
              The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
              token instead.
          pad_token (`str`, *optional*, defaults to `"<|pad|>"`):
              The token used for padding, for example when batching sequences of different lengths.
          sp_model_kwargs (`dict`, *optional*):
              Will be passed to the `SentencePieceProcessor.__init__()` method. The [Python wrapper for
              SentencePiece](https://github.com/google/sentencepiece/tree/master/python) can be used, among other things,
              to set:
              - `enable_sampling`: Enable subword regularization.
              - `nbest_size`: Sampling parameters for unigram. Invalid for BPE-Dropout.
                - `nbest_size = {0,1}`: No sampling is performed.
                - `nbest_size > 1`: samples from the nbest_size results.
                - `nbest_size < 0`: assuming that nbest_size is infinite and samples from the all hypothesis (lattice)
                  using forward-filtering-and-backward-sampling algorithm.
              - `alpha`: Smoothing parameter for unigram sampling, and dropout probability of merge operations for
                BPE-dropout.
      """

    def __init__(self, vocab_file: str = 'spiece.model', unk_token='<|unk|>', pad_token='<|pad|>',
                 bos_token="<extra_id_0>", eos_token='<extra_id_2>', sep_token='<extra_id_1>',
                 sp_model_kwargs: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs
        super().__init__(bos_token=bos_token, eos_token=eos_token, unk_token=unk_token,
                         pad_token=pad_token, sep_token=sep_token,
                         sp_model_kwargs=self.sp_model_kwargs, **kwargs)
        self.vocab_file = vocab_file
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(vocab_file)

    @property
    def vocab_size(self):
        return self.sp_model.get_piece_size()

    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def __getstate__(self):
        state = self.__dict__.copy()
        state['sp_model'] = None
        return state

    def __setstate__(self, d):
        self.__dict__ = d
        if not hasattr(self, 'sp_model_kwargs'):
            self.sp_model_kwargs = {}
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.load(self.vocab_file)

    def _tokenize(self, text: str) -> List[str]:
        """Take as input a string and return a list of strings (tokens) for words/sub-words"""
        return self.sp_model.encode(text, out_type=str)

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        token = self.sp_model.id_to_piece(index)
        return token

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        return self.sp_model.decode(tokens)
