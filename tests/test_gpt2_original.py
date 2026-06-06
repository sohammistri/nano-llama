"""Tests for the GPT-2 original architecture."""

import torch

from nanollama.models import MODEL_REGISTRY, build_model_from_config
from nanollama.models.gpt2_original import GPT2Original, GPT2OriginalConfig


def _tiny_config(**overrides):
    kwargs = dict(
        sequence_len=16,
        vocab_size=128,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=32,
        embd_pdrop=0.0,
        resid_pdrop=0.0,
        attn_pdrop=0.0,
    )
    kwargs.update(overrides)
    return GPT2OriginalConfig(**kwargs)


# [CRITICAL] Validates registry discovery and model construction contracts.
def test_gpt2_original_is_registered():
    assert "gpt2_original" in MODEL_REGISTRY
    config, model_cls = build_model_from_config("gpt2_original", _tiny_config().__dict__)
    assert config.arch == "gpt2_original"
    assert model_cls is GPT2Original


# [CRITICAL] Forward/loss shapes guard against architecture wiring mistakes.
def test_gpt2_original_forward_and_loss_shapes():
    model = GPT2Original(_tiny_config())
    model.init_weights()
    model.eval()
    idx = torch.randint(0, model.config.vocab_size, (3, 8))
    targets = torch.randint(0, model.config.vocab_size, (3, 8))

    logits = model(idx)
    loss = model(idx, targets)

    assert logits.shape == (3, 8, model.config.vocab_size)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


# [HIGH] Ensures the default GPT-2 LM head uses tied token embeddings.
def test_gpt2_original_ties_lm_head_by_default():
    model = GPT2Original(_tiny_config())
    model.init_weights()

    assert model.lm_head.weight is model.transformer.wte.weight
    assert model.num_scaling_params()["lm_head"] == 0


# [CRITICAL] Catches silent optimizer parameter duplication or missed params.
def test_gpt2_original_optimizer_groups_cover_unique_parameters_once():
    model = GPT2Original(_tiny_config())
    model.init_weights()
    param_groups = model._get_optimizer_param_groups(
        unembedding_lr=0.004,
        embedding_lr=0.2,
        matrix_lr=0.02,
        weight_decay=0.1,
        scalar_lr=0.5,
    )

    grouped = [param for group in param_groups for param in group["params"]]
    assert len(grouped) == len({id(param) for param in grouped})
    assert {id(param) for param in grouped} == {id(param) for param in model.parameters()}
    assert all(group["kind"] == "adamw" for group in param_groups)


# [HIGH] Confirms untied output heads are represented in scaling counts and optimizer groups.
def test_gpt2_original_can_disable_weight_tying():
    model = GPT2Original(_tiny_config(tie_word_embeddings=False))
    model.init_weights()

    assert model.lm_head.weight is not model.transformer.wte.weight
    assert model.num_scaling_params()["lm_head"] == model.lm_head.weight.numel()