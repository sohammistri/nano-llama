# CLAUDE.md — Engineering Guidelines for nano-llama

This file documents conventions and rules that Claude Code (and any external engineer) must follow when working in this repository.

---

## Python Execution: Always Use `uv run`

**All Python commands must be prefixed with `uv run`.** This project uses [uv](https://github.com/astral-sh/uv) for dependency and environment management. Never invoke `python`, `python3`, or `torchrun` directly — always go through `uv run` to ensure the correct virtual environment and dependency versions are used.

```bash
# Correct
uv run python -m scripts.chat_eval_open_router -m meta-llama/llama-3.1-8b-instruct -a GSM8K
uv run torchrun --nproc_per_node=8 -m scripts.base_eval --hf-path openai-community/gpt2
uv run python -m pytest

# Wrong — do not use these forms
python -m scripts.chat_eval_open_router ...
python3 scripts/base_eval.py ...
torchrun ...
```

This applies to all subcommands: running scripts, running tests, installing tools, or invoking any CLI that lives in the project's virtualenv.

---

## Temporary Files: Use `tmp/`

Any temporary or scratch files created during development — one-off scripts, debug outputs, quick experiments, intermediate data files — must be placed in the `tmp/` directory at the repo root.

```
tmp/
├── debug_response.json       # example: a raw API response you dumped for inspection
├── scratch_eval.py           # example: a throwaway script to test something quickly
└── ...
```

**Rationale**: keeps the repo clean and makes it obvious what is real code vs. throwaway work. The `tmp/` directory is git-ignored.

Do not scatter temporary files in `scripts/`, `notebooks/`, or the repo root.

---

## Repository Overview

```
nano-llama/
├── nanollama/              # Core library (models, engine, tokenizer, eval utilities)
├── tasks/                  # Chat eval task definitions for local/HF models
├── open_router_tasks/      # Chat eval task definitions for OpenRouter API
├── scripts/                # Entry-point scripts (eval, sweep, download)
├── notebooks/              # Exploratory scripts and notebooks
├── tmp/                    # Temporary scratch files (git-ignored)
├── pyproject.toml          # Project dependencies (managed via uv)
└── uv.lock                 # Locked dependency versions
```

### Key entry points

| Script | Purpose |
|--------|---------|
| `scripts/base_eval.py` | Evaluate base (non-chat) models: CORE ICL accuracy, bits-per-byte, sampling |
| `scripts/chat_eval.py` | Evaluate chat models locally or via HuggingFace |
| `scripts/chat_eval_open_router.py` | Evaluate any OpenRouter model on standard benchmarks |
| `scripts/download_hf_model.py` | Pre-cache a HuggingFace model for offline use |
| `scripts/run_eval_sweep_hf.sh` | Sweep CORE eval across multiple HF models |
| `scripts/run_eval_sweep_open_router.sh` | Sweep all chat benchmarks across multiple OpenRouter models |

### Benchmarks

**Base model (local, via `base_eval.py`):** CORE ICL tasks, bits-per-byte on train/val data.

**Chat model — local (`chat_eval.py`):** ARC-Easy, ARC-Challenge, MMLU, GSM8K, HumanEval, SpellingBee.

**Chat model — OpenRouter (`chat_eval_open_router.py`):** GSM8K, MATH500, GPQA Diamond, ARC Challenge, MBPP+, HumanEval+.

---

## Environment Variables

| Variable | Used by | Description |
|----------|---------|-------------|
| `OPENROUTER_API_KEY` | `open_router_tasks/`, `scripts/chat_eval_open_router.py` | OpenRouter API key |
| `HF_TOKEN` | HuggingFace model loading | Access token for gated models |
| `NANOCHAT_BASE_DIR` | `scripts/base_eval.py` | Output directory for results (default: `out/`) |

---

## Testing

```bash
uv run python -m pytest
```

---

## Adding a New OpenRouter Benchmark

1. Create `open_router_tasks/<benchmark_name>.py` subclassing `BaseOpenRouterTask` from `open_router_tasks/common.py`.
2. Implement `build_messages(row)`, `_eval_single(i)`, and `debug_single(seed=None)`.
3. Register the new class in `scripts/chat_eval_open_router.py` under the `task_map` dict.
4. Add the task name to the `ALL` expansion list in the same script.

## Adding a New Local Chat Benchmark

1. Create `tasks/<benchmark_name>.py` subclassing `Task` from `tasks/common.py`.
2. Implement `eval_type`, `num_examples()`, `get_example(index)`, and `evaluate(conversation, completion)`.
3. Import and register it in `scripts/chat_eval.py`.
