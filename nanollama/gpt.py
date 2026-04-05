"""
Backward-compatibility shim: re-exports from nanochat.models.gpt and nanochat.models.base.
All existing imports (e.g. `from nanochat.gpt import GPT, GPTConfig, Linear`) continue to work.
"""

from nanochat.models.gpt import GPT, GPTConfig, CausalSelfAttention, MLP, Block, has_ve  # noqa: F401
from nanochat.models.base import Linear, norm, apply_rotary_emb  # noqa: F401
