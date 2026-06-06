"""
Faithful GPT-2 style decoder-only Transformer.

This architecture intentionally avoids the custom nano-llama GPT features
(RoPE, sliding windows, QK norm, value embeddings, smear, backout, logit
softcap, and ReLU^2 MLPs) so it can serve as a plain GPT-2 baseline.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanollama.common import COMPUTE_DTYPE
from nanollama.models.base import BaseLM, BaseLMConfig, Linear


@dataclass
class GPT2OriginalConfig(BaseLMConfig):
    arch: str = "gpt2_original"
    pos_emb_type: str = "learned"
    embd_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    bias: bool = True
    tie_word_embeddings: bool = True


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        assert config.n_kv_head == config.n_head, "GPT-2 original uses regular multi-head attention, not GQA"
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.attn_pdrop = config.attn_pdrop
        self.c_attn = Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def _attention(self, q, k, v, kv_cache):
        if kv_cache is None:
            dropout_p = self.attn_pdrop if self.training else 0.0
            return F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=dropout_p)

        Tq = q.size(2)
        Tk = k.size(2)
        if Tq == Tk:
            dropout_p = self.attn_pdrop if self.training else 0.0
            return F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=dropout_p)
        if Tq == 1:
            return F.scaled_dot_product_attention(q, k, v, is_causal=False, dropout_p=0.0)

        device = q.device
        row_idx = (Tk - Tq) + torch.arange(Tq, device=device).unsqueeze(1)
        col_idx = torch.arange(Tk, device=device).unsqueeze(0)
        mask = col_idx <= row_idx
        dropout_p = self.attn_pdrop if self.training else 0.0
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout_p)

    def forward(self, x, kv_cache=None, layer_idx=None):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if kv_cache is not None:
            k_cache, v_cache = kv_cache.get_layer_cache(layer_idx)
            pos = kv_cache.get_pos()
            k_cache[:, pos:pos + T, :, :] = k.transpose(1, 2)
            v_cache[:, pos:pos + T, :, :] = v.transpose(1, 2)
            k = k_cache[:, :pos + T, :, :].transpose(1, 2)
            v = v_cache[:, :pos + T, :, :].transpose(1, 2)

        y = self._attention(q, k, v, kv_cache)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        if kv_cache is not None and layer_idx == kv_cache.n_layers - 1:
            kv_cache.advance(T)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x, approximate="tanh")
        x = self.c_proj(x)
        return self.dropout(x)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, elementwise_affine=True, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, elementwise_affine=True, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, kv_cache=None, layer_idx=None):
        x = x + self.attn(self.ln_1(x), kv_cache=kv_cache, layer_idx=layer_idx)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2Original(BaseLM):
    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__(config, pad_vocab_size_to)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.ln_f = nn.LayerNorm(config.n_embd, elementwise_affine=True, bias=config.bias)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.transformer.wte.weight

    def _build_blocks(self, config):
        return nn.ModuleList([Block(config) for _ in range(config.n_layer)])

    @torch.no_grad()
    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.ones_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

        for name, param in self.named_parameters():
            if name.endswith("c_proj.weight"):
                torch.nn.init.normal_(param, mean=0.0, std=0.02 / (2 * self.config.n_layer) ** 0.5)

        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.transformer.wte.weight
        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)
            self.position_embeddings.to(dtype=COMPUTE_DTYPE)

    def _transformer_forward(self, x, x0, cos_sin, kv_cache, idx=None):
        for layer_idx, block in enumerate(self.transformer.h):
            x = block(x, kv_cache=kv_cache, layer_idx=layer_idx)
        return x

    def _non_matmul_param_numel(self):
        total = 0
        for name, param in self.named_parameters():
            if name in {"transformer.wte.weight", "position_embeddings.weight"}:
                total += param.numel()
            elif param.ndim < 2:
                total += param.numel()
        return total

    def num_scaling_params(self):
        wte = self.transformer.wte.weight.numel()
        position_embeddings = self.position_embeddings.weight.numel()
        lm_head = 0 if self.config.tie_word_embeddings else self.lm_head.weight.numel()
        transformer_matrices = 0
        layernorm_and_biases = 0
        for name, param in self.named_parameters():
            if name in {"transformer.wte.weight", "position_embeddings.weight", "lm_head.weight"}:
                continue
            if param.ndim >= 2:
                transformer_matrices += param.numel()
            else:
                layernorm_and_biases += param.numel()
        total = wte + position_embeddings + lm_head + transformer_matrices + layernorm_and_biases
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            "wte": wte,
            "position_embeddings": position_embeddings,
            "lm_head": lm_head,
            "transformer_matrices": transformer_matrices,
            "layernorm_and_biases": layernorm_and_biases,
            "total": total,
        }

    def _get_optimizer_param_groups(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, scalar_lr=0.5):
        decay_params = []
        no_decay_params = []
        embedding_params = []
        lm_head_params = []

        seen = set()
        for name, param in self.named_parameters():
            if id(param) in seen:
                continue
            seen.add(id(param))
            if name == "transformer.wte.weight" or name == "position_embeddings.weight":
                embedding_params.append(param)
            elif name == "lm_head.weight":
                lm_head_params.append(param)
            elif param.ndim >= 2:
                decay_params.append(param)
            else:
                no_decay_params.append(param)

        param_groups = []
        if decay_params:
            param_groups.append(dict(kind="adamw", params=decay_params, lr=matrix_lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=weight_decay))
        if no_decay_params:
            param_groups.append(dict(kind="adamw", params=no_decay_params, lr=matrix_lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0))
        if embedding_params:
            param_groups.append(dict(kind="adamw", params=embedding_params, lr=embedding_lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0))
        if lm_head_params:
            param_groups.append(dict(kind="adamw", params=lm_head_params, lr=unembedding_lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=weight_decay))
        return param_groups

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction="mean"):
        B, T = idx.size()
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        assert T0 + T <= self.config.sequence_len, f"Sequence length grew beyond configured context: {T0 + T} > {self.config.sequence_len}"

        positions = torch.arange(T0, T0 + T, device=idx.device)
        x = self.transformer.wte(idx).to(COMPUTE_DTYPE)
        x = x + self.position_embeddings(positions).to(x.dtype)
        x = self.drop(x)
        x = self._transformer_forward(x, x, None, kv_cache, idx=idx)
        x = self.ln_f(x)

        logits = self.lm_head(x)
        logits = logits[..., :self.config.vocab_size]
        logits = logits.float()

        if targets is not None:
            return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
        return logits


from nanollama.models import register_model
register_model("gpt2_original", GPT2OriginalConfig, GPT2Original)