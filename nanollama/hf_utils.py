"""
Shared utilities for loading and wrapping HuggingFace models.
Used by both scripts/base_eval.py and scripts/chat_eval.py.
"""

import torch
from nanollama.common import print0
from nanollama.tokenizer import HuggingFaceTokenizer


class ModelWrapper:
    """Lightweight wrapper to give HuggingFace models a nanollama-compatible interface."""
    def __init__(self, model, max_seq_len=None):
        self.model = model
        self.max_seq_len = max_seq_len

    def __call__(self, input_ids, targets=None, loss_reduction='mean'):
        logits = self.model(input_ids).logits
        if targets is None:
            return logits
        B, T, V = logits.shape
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, V),
            targets.view(-1),
            ignore_index=-1,
            reduction=loss_reduction
        )
        if loss_reduction == 'none':
            loss = loss.view(B, T)
        return loss

    def get_device(self):
        return next(self.model.parameters()).device

    def generate(self, input_ids, **kwargs):
        """Delegate to the underlying HF model's generate() method."""
        return self.model.generate(input_ids, **kwargs)


def load_hf_model(hf_path: str, device, device_map=None, quantize=None):
    """Load a HuggingFace model and tokenizer.

    Returns (ModelWrapper, HuggingFaceTokenizer) where the tokenizer has
    a transformers AutoTokenizer attached for chat template support.

    When device_map is set (e.g. "auto"), the model is sharded across
    multiple GPUs for large models that don't fit on a single device.

    When quantize is set ("4bit" or "8bit"), the model is loaded with
    bitsandbytes quantization. This implies device_map="auto".
    """
    print0(f"Loading HuggingFace model from: {hf_path}")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    load_kwargs = dict(
        torch_dtype=torch.bfloat16,
        device_map=device_map or device,
    )

    if quantize:
        from transformers import BitsAndBytesConfig
        # Quantization requires device_map
        load_kwargs["device_map"] = device_map or "auto"
        if quantize == "4bit":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
        elif quantize == "8bit":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        else:
            raise ValueError(f"Unsupported quantize value: {quantize}. Use '4bit' or '8bit'.")
        # Don't set torch_dtype when quantizing — the config handles precision
        del load_kwargs["torch_dtype"]
        print0(f"Using {quantize} quantization via bitsandbytes")

    model = AutoModelForCausalLM.from_pretrained(hf_path, **load_kwargs)
    model.eval()
    # Set max_seq_len from model config, capped for eval workloads
    if "gpt2" in hf_path:
        max_seq_len = 1024
    else:
        max_seq_len = getattr(model.config, 'max_position_embeddings', 8192)
        max_seq_len = min(max_seq_len, 8192)
    model = ModelWrapper(model, max_seq_len=max_seq_len)
    tokenizer = HuggingFaceTokenizer.from_pretrained(hf_path)
    # Also load the transformers AutoTokenizer for apply_chat_template support
    auto_tokenizer = AutoTokenizer.from_pretrained(hf_path)
    tokenizer.set_auto_tokenizer(auto_tokenizer)
    return model, tokenizer
