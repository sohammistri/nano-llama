"""
Base LLM class and shared utilities for all model architectures.

Shared utilities: norm, Linear, apply_rotary_emb
Base classes: BaseLMConfig, BaseLM
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanollama.common import get_dist_info, print0, COMPUTE_DTYPE
from nanollama.optim import MuonAdamW, DistMuonAdamW, PureAdamW


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def norm(x):
    return F.rms_norm(x, (x.size(-1),))  # note that this will run in bf16, seems ok


class Linear(nn.Linear):
    """nn.Linear that casts weights to match input dtype in forward.
    Replaces autocast: master weights stay fp32 for optimizer precision,
    but matmuls run in the activation dtype (typically bf16 from embeddings)."""
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]  # split up last dim into two halves
    y1 = x1 * cos + x2 * sin  # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


# ---------------------------------------------------------------------------
# MoE building blocks
# ---------------------------------------------------------------------------

class TopKRouter(nn.Module):
    """DeepSeekV3-style sigmoid routing with bias nudging (no auxiliary loss needed).
    Routes each token to top_k experts based on sigmoid scores + learnable bias."""

    def __init__(self, n_embd, num_experts, top_k=2):
        super().__init__()
        self.top_k = top_k
        self.gate = Linear(n_embd, num_experts, bias=False)
        self.register_buffer("expert_bias", torch.zeros(num_experts))

    def forward(self, x):
        """Route tokens to experts.
        Args:
            x: (B, T, D) input tensor
        Returns:
            top_k_indices: (B, T, top_k) selected expert indices
            weights: (B, T, top_k) normalized routing weights
            aux_loss: scalar (always 0 — bias nudging replaces aux loss)
        """
        scores = torch.sigmoid(self.gate(x))
        biased = scores + self.expert_bias.detach()
        top_k_scores, top_k_indices = torch.topk(biased, self.top_k, dim=-1)
        weights = torch.gather(scores, -1, top_k_indices)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-6)
        return top_k_indices, weights, torch.tensor(0.0, device=x.device)

    @torch.no_grad()
    def update_bias(self, expert_counts, target_count, nudge_rate=0.001):
        """Nudge expert biases to balance load. Call after each batch."""
        self.expert_bias += nudge_rate * (target_count - expert_counts.float())


class MoEMLP(nn.Module):
    """Mixture-of-Experts MLP layer. Routes tokens to top-k expert MLPs
    with optional shared expert.

    Args:
        n_embd: model dimension
        num_experts: number of routed experts
        top_k: number of experts per token
        shared_expert: whether to add a shared (always-on) expert
        mlp_factory: callable that returns an MLP module
    """

    def __init__(self, n_embd, num_experts, top_k=2, shared_expert=True, mlp_factory=None):
        super().__init__()
        assert mlp_factory is not None, "mlp_factory is required"
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = TopKRouter(n_embd, num_experts, top_k)
        self.experts = nn.ModuleList([mlp_factory() for _ in range(num_experts)])
        self.shared_expert = mlp_factory() if shared_expert else None

    def forward(self, x):
        """
        Args:
            x: (B, T, D) input tensor
        Returns:
            output: (B, T, D) mixture output
            aux_loss: scalar auxiliary loss from router
        """
        B, T, D = x.shape
        expert_indices, expert_weights, aux_loss = self.router(x)

        # Flatten for dispatch
        flat_x = x.view(-1, D)  # (B*T, D)
        flat_indices = expert_indices.view(-1, self.top_k)  # (B*T, top_k)
        flat_weights = expert_weights.view(-1, self.top_k)  # (B*T, top_k)

        # Loop-based dispatch (simple baseline)
        output = torch.zeros_like(flat_x)
        for k in range(self.top_k):
            for e in range(self.num_experts):
                mask = (flat_indices[:, k] == e)
                if mask.any():
                    output[mask] += flat_weights[mask, k:k+1] * self.experts[e](flat_x[mask])

        output = output.view(B, T, D)
        if self.shared_expert is not None:
            output = output + self.shared_expert(x)
        return output, aux_loss


# ---------------------------------------------------------------------------
# Base config and model
# ---------------------------------------------------------------------------

@dataclass
class BaseLMConfig:
    """Minimal config fields shared by all architectures.
    Subclasses add architecture-specific fields."""
    arch: str = "gpt"           # Registry key for model lookup
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6             # number of query heads
    n_kv_head: int = 6          # number of key/value heads (GQA)
    n_embd: int = 768
    pos_emb_type: str = "rope"  # "rope" | "learned" | "none"


class BaseLM(nn.Module):
    """
    Base language model providing shared infrastructure:
    - Token embedding (wte) and lm_head
    - Rotary embedding precomputation + buffers
    - get_device(), generate()
    - setup_optimizer() framework (delegates param groups to subclass)
    - estimate_flops() (delegates non-matmul exclusions to subclass)
    - forward() template: embed → norm → _transformer_forward() → norm → lm_head → loss

    Subclasses must override:
    - _build_blocks(config) → nn.ModuleList
    - init_weights()
    - _transformer_forward(x, x0, cos_sin, kv_cache) → x
    - _get_optimizer_param_groups(**kwargs) → list[dict]
    - _non_matmul_param_numel() → int
    - num_scaling_params() → dict
    """

    def __init__(self, config, pad_vocab_size_to=64):
        """
        NOTE: this __init__ runs in meta device context.
        Any calculations inside here are shapes and dtypes only, no actual data.
        We actually initialize all data in init_weights() instead.
        """
        super().__init__()
        self.config = config
        # Pad vocab for efficiency (DDP, tensor cores)
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        self.padded_vocab_size = padded_vocab_size

        # Build transformer: wte + architecture-specific blocks
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": self._build_blocks(config),
        })
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)

        # Position embeddings (conditional on config)
        if config.pos_emb_type == "rope":
            self.rotary_seq_len = config.sequence_len * 10
            head_dim = config.n_embd // config.n_head
            cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)
        elif config.pos_emb_type == "learned":
            self.position_embeddings = nn.Embedding(config.sequence_len, config.n_embd)
        elif config.pos_emb_type == "none":
            pass
        else:
            raise ValueError(f"Unknown pos_emb_type: {config.pos_emb_type}")

        # Auxiliary loss accumulator (used by MoE architectures, not by base forward)
        self.aux_loss = torch.tensor(0.0)

    def _build_blocks(self, config):
        """Construct transformer blocks. Must be overridden by subclasses."""
        raise NotImplementedError

    def init_weights(self):
        """Architecture-specific weight initialization. Must be overridden."""
        raise NotImplementedError

    def _transformer_forward(self, x, x0, cos_sin, kv_cache):
        """Core block loop. Must be overridden by subclasses."""
        raise NotImplementedError

    def _get_optimizer_param_groups(self, **kwargs):
        """Return list of param group dicts for the optimizer. Must be overridden."""
        raise NotImplementedError

    def _non_matmul_param_numel(self):
        """Return count of params to exclude from FLOPs (embeddings, scalars). Must be overridden."""
        raise NotImplementedError

    def num_scaling_params(self):
        """Return parameter breakdown dict for scaling laws. Must be overridden."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Shared infrastructure
    # ------------------------------------------------------------------

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=100000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.to(COMPUTE_DTYPE), sin.to(COMPUTE_DTYPE)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def get_device(self):
        return self.transformer.wte.weight.device

    def _prepare_position_info(self, idx, T, kv_cache):
        """Prepare position-related tensors based on pos_emb_type."""
        if self.config.pos_emb_type == "rope":
            assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
            assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
            assert self.cos.dtype == COMPUTE_DTYPE, f"Rotary embeddings must be in {COMPUTE_DTYPE}, got {self.cos.dtype}"
            T0 = 0 if kv_cache is None else kv_cache.get_pos()
            return self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]
        return None

    def estimate_flops(self):
        """
        Return the estimated FLOPs per token for the model (forward + backward).
        Each matmul weight parameter contributes 6 FLOPs (2 forward + 4 backward).
        Attention FLOPs: 12 * h * q * effective_seq_len per layer.
        """
        nparams = sum(p.numel() for p in self.parameters())
        nparams_exclude = self._non_matmul_param_numel()
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        attn_flops = self._compute_attn_flops(h, q, t)
        num_flops_per_token = 6 * (nparams - nparams_exclude) + attn_flops
        return num_flops_per_token

    def _compute_attn_flops(self, h, q, t):
        """Compute attention FLOPs. Override if architecture has window patterns."""
        return self.config.n_layer * 12 * h * q * t

    def setup_optimizer(self, **kwargs):
        """Build optimizer using subclass-provided param groups.
        Auto-selects optimizer factory: PureAdamW when no Muon groups exist,
        MuonAdamW/DistMuonAdamW otherwise."""
        ddp, rank, local_rank, world_size = get_dist_info()
        param_groups = self._get_optimizer_param_groups(**kwargs)
        has_muon = any(g.get('kind') == 'muon' for g in param_groups)
        if has_muon:
            Factory = DistMuonAdamW if ddp else MuonAdamW
        else:
            Factory = DistMuonAdamW if ddp else PureAdamW
        optimizer = Factory(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        B, T = idx.size()

        # Position info (rope cos/sin or None)
        pos_info = self._prepare_position_info(idx, T, kv_cache)

        # Embed tokens
        x = self.transformer.wte(idx)
        x = x.to(COMPUTE_DTYPE)

        # Add learned position embeddings if configured
        if self.config.pos_emb_type == "learned":
            T0 = 0 if kv_cache is None else kv_cache.get_pos()
            positions = torch.arange(T0, T0 + T, device=idx.device)
            x = x + self.position_embeddings(positions).to(x.dtype)

        x = norm(x)

        # Architecture-specific transformer forward
        x0 = x
        x = self._transformer_forward(x, x0, pos_info, kv_cache, idx=idx)
        x = norm(x)

        # lm_head → logits → loss
        logits = self.lm_head(x)
        logits = logits[..., :self.config.vocab_size]
        logits = logits.float()

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            return loss
        else:
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """Naive autoregressive streaming inference (batch size 1)."""
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        for _ in range(max_tokens):
            logits = self.forward(ids)
            logits = logits[:, -1, :]
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
