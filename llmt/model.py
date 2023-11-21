import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Transformer(nn.Module):
    """
    Transformer model.
    """

    def __init__(self, vocab_size: int, layers: int, n_heads: int, embeddings_size: int,
                 dropout: float, bias: bool, context_size: int,
                 focus_temp: float, focus_percent: float, focus_min_seq_len: int,
                 focus_num_fixed_positions: int):
        """
        Initializes the transformer model.
        :param vocab_size: the size of the vocabulary
        :param layers: the number of layers
        :param n_heads: the number of attention heads
        :param embeddings_size:  the size of the embeddings for all attention heads
        :param dropout: the dropout rate
        :param bias: whether to use bias or not
        :param context_size: the context size or maximum length of the sequences
        :param focus_temp: the temperature to use for the softmax for focus attention
        :param focus_percent: the percentage of tokens to focus on the layers
        :param focus_min_seq_len: the minimum sequence length to apply focus attention
        :param focus_num_fixed_positions: the number of fixed positions to add in focus attention
        """
        super().__init__()
        self.layers = layers
        self.vocab_size = vocab_size
        self.n_heads = n_heads
        self.embeddings_size = embeddings_size
        self.dropout = dropout
        self.context_size = context_size
        self.focus_num_fixed_positions = focus_num_fixed_positions

        self.vocab_embed = nn.Embedding(vocab_size, embeddings_size)
        self.positional_embed = nn.Embedding(context_size, embeddings_size)
        self.dropout = nn.Dropout(dropout)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(n_heads, embeddings_size, dropout, bias,
                              focus_temp, focus_percent, focus_min_seq_len) for _ in range(layers)])
        self.layer_normalization = nn.LayerNorm(embeddings_size)

        self.decoder = nn.Linear(embeddings_size, vocab_size)
        # Tie the weights of the embedding layer and the pre-softmax linear transformation
        self.decoder.weight = self.vocab_embed.weight

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.layers))

    def get_num_params(self) -> int:
        """
        :return:  the number of parameters of the model.
        """
        n_params = sum(p.numel() for p in self.parameters())
        # subtract the tie weights for the word embeddings and the decoder
        n_params -= self.decoder.weight.numel()
        return n_params

    def _init_weights(self, module):
        """
        Initializes the weights of the model.
        :param module: the module to initialize the weights for
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, sequences, targets=None, mask=None, sep_token=None) \
            -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of the model.
        :param sequences: the sequences to use for the predictions
        :param targets: the targets to use for the loss calculation
        :param mask: the mask to use for the loss calculation
        :param sep_token: the serparator token to use for the loss calculation when finetuning the
        model
        :return: a tuple of the logits and the loss (if targets are given)
        """
        device = sequences.device
        b, t = sequences.size()
        assert t <= self.context_size, \
            f"Cannot forward sequence of length {t}, context size is only {self.context_size}"

        pos = torch.arange(0, t, dtype=torch.int32, device=device)  # shape (t)

        if mask is not None and sep_token is not None:
            # do not use the initial text for loss calculation
            mask_aux = torch.cumsum(sequences == sep_token, dim=-1)
            mask = mask * mask_aux

        # forward the GPT model itself
        tok_emb = self.vocab_embed(sequences)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.positional_embed(pos)  # position embeddings of shape (t, n_embd)
        x = self.dropout(tok_emb + pos_emb)
        if self.training:
            fixed_positions = torch.randint(low=0, high=t,
                                            size=(b, self.focus_num_fixed_positions),
                                            device=x.device)
            fixed_pos_masks = torch.zeros((b, t), dtype=torch.bool, device=x.device)
            batch_indices = torch.arange(b, device=x.device)
            batch_indices = batch_indices.view(-1, 1).expand_as(fixed_positions)
            fixed_pos_masks[batch_indices, fixed_positions] = True
        else:
            fixed_pos_masks = None
        for block in self.transformer_blocks:
            casual_mask = nn.Transformer.generate_square_subsequent_mask(x.size(1),
                                                                         device=device,
                                                                         dtype=torch.bool)
            x, targets, mask, fixed_pos_masks = block(x, is_casual=True, attn_mask=casual_mask,
                                                      targets=targets, mask=mask,
                                                      fixed_positions=fixed_pos_masks)
        x = self.layer_normalization(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.decoder(x)
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = targets.contiguous().view(-1)

            if mask is not None:
                mask_flat = mask.contiguous().view(-1).to(dtype=torch.bool)
                logits_flat = logits_flat[mask_flat]
                targets_flat = targets_flat[mask_flat]
            loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.decoder(x[:, [-1], :])  # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(self, sequences: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                 top_k: Optional[int] = None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.

        :param sequences: the sequences to use as context for the generation
        :param max_new_tokens: the maximum number of tokens to generate
        :param temperature: the temperature to use for the softmax
        :param top_k: the number of top tokens to consider for sampling
        :return: the generated tokens
        """
        if top_k is not None:
            assert top_k > 0
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            sequences_cropped = sequences
            if sequences.size(1) > self.context_size:
                sequences_cropped = sequences[:, -self.context_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(sequences_cropped)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            sequences = torch.cat((sequences, idx_next), dim=1)

        return sequences


class TransformerBlock(nn.Module):
    """
    A transformer block as used in GPT models.
    """

    def __init__(self, n_heads: int, n_embeddings: int, dropout: float, bias: bool,
                 focus_temperature: float, focus_percent: float, focus_min_seq_len: int):
        """
        :param n_heads: the number of heads to use in the multihead attention
        :param n_embeddings: the number of embeddings
        :param dropout: the dropout probability
        :param bias: whether to use bias in the multihead attention
        :param focus_temperature: the temperature to use for the softmax for focus attention
        :param focus_percent: the percentage of tokens to focus on the layer
        :param focus_min_seq_len: the minimum sequence length to apply focus attention
        """
        super().__init__()
        assert n_embeddings % n_heads == 0, \
            f"n_embeddings ({n_embeddings}) must be divisible by n_heads ({n_heads})"
        assert 0 <= focus_percent <= 1, "focus_percent must be between 0 and 1"
        assert 0 <= focus_temperature <= 1, "focus_temperature must be between 0 and 1"
        assert focus_min_seq_len > 0, "focus_min_seq_len must be greater than 0"
        self.focus_temperature = focus_temperature
        self.focus_percent = focus_percent
        self.focus_min_seq_len = focus_min_seq_len
        self.ln_1 = nn.LayerNorm(n_embeddings)
        self.multiheadattention = nn.MultiheadAttention(embed_dim=n_embeddings, num_heads=n_heads,
                                                        dropout=dropout, bias=bias,
                                                        batch_first=True)
        self.ln_2 = nn.LayerNorm(n_embeddings)
        self.mlp = MLP(n_embeddings, dropout, bias)

    def forward(self, x, is_casual, attn_mask, targets=None, mask=None, fixed_positions=None):
        qkv = self.ln_1(x)
        attention, weights = self.multiheadattention(qkv, qkv, qkv,
                                                     is_causal=is_casual, attn_mask=attn_mask,
                                                     average_attn_weights=True)
        seq_len = x.shape[1]
        if seq_len > self.focus_min_seq_len:
            next_seq_len = math.floor(seq_len * self.focus_percent)
            sum_values = torch.sum(weights, dim=1)
            n = torch.arange(sum_values.shape[1], 0, -1, device=sum_values.device)
            token_contribution = sum_values / n

            if self.training:
                # set probability of 1 to fixed positions and select the indices
                token_contribution = F.softmax(token_contribution / self.focus_temperature, dim=-1)
                token_contribution[fixed_positions] = 1.0
                idx_next = torch.multinomial(token_contribution, num_samples=next_seq_len)
            else:
                # select indices of the top k tokens
                _, idx_next = torch.topk(token_contribution, k=next_seq_len, dim=1, sorted=False)
            idx_next, _ = idx_next.sort(dim=1)
            # Adjust indices for gather
            indices = idx_next.unsqueeze(-1).expand(-1, -1, attention.shape[-1])
            # Focus attention on the selected tokens
            attention = torch.gather(attention, 1, indices)
            x = torch.gather(x, 1, indices)
            if mask is not None:
                mask = torch.gather(mask, 1, idx_next)
            if targets is not None:
                targets = torch.gather(targets, 1, idx_next)
            if fixed_positions is not None:
                fixed_positions = torch.gather(fixed_positions, 1, idx_next)

        x = x + attention
        x = x + self.mlp(self.ln_2(x))
        return x, targets, mask, fixed_positions


class MLP(nn.Module):
    """
    A multy layer perceptron used in GPT models: a linear layer + gelu + linear + dropout.
    """

    def __init__(self, n_embeddings: int, dropout: float, bias: bool):
        """
        :param n_embeddings: the number of embeddings
        :param dropout: the dropout probability
        :param bias: whether to use bias in the linear layers
        """
        super().__init__()
        self.c_fc = nn.Linear(n_embeddings, 4 * n_embeddings, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embeddings, n_embeddings, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
