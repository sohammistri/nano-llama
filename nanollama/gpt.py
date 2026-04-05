"""
Backward-compatibility shim: re-exports from nanollama.models.gpt and nanollama.models.base.
All existing imports (e.g. `from nanollama.gpt import GPT, GPTConfig, Linear`) continue to work.
"""

from nanollama.models.gpt import GPT, GPTConfig, CausalSelfAttention, MLP, Block, has_ve  # noqa: F401
from nanollama.models.base import Linear, norm, apply_rotary_emb  # noqa: F401
