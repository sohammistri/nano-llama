# nano-llama

A research framework for training, evaluating, and benchmarking language models — from scratch-trained nano-LLaMA variants to off-the-shelf HuggingFace and OpenRouter-hosted models.

---

## Table of Contents

- [Setup](#setup)
- [Repository Structure](#repository-structure)
- [Base Model Evaluation](#base-model-evaluation)
- [Chat Model Evaluation (Local / HuggingFace)](#chat-model-evaluation-local--huggingface)
- [Chat Model Evaluation (OpenRouter API)](#chat-model-evaluation-openrouter-api)
- [Benchmark Reference](#benchmark-reference)
- [Model Sweep Scripts](#model-sweep-scripts)
- [Results & Logs](#results--logs)

---

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies
uv sync
```

For OpenRouter-based evals, export your API key:

```bash
export OPENROUTER_API_KEY=sk-or-...
```

For gated HuggingFace models (e.g. LLaMA), export your HF token:

```bash
export HF_TOKEN=hf_...
```

---

## Repository Structure

```
nano-llama/
├── nanollama/              # Core library: model, training engine, tokenizer, eval utilities
│   ├── models/             # Model architecture definitions (GPT/LLaMA variants)
│   ├── engine.py           # Inference engine (batched generation)
│   ├── checkpoint_manager.py
│   ├── core_eval.py        # ICL (in-context learning) eval loop
│   ├── loss_eval.py        # Bits-per-byte evaluation
│   └── hf_utils.py         # HuggingFace model loading wrappers
│
├── tasks/                  # Local chat eval tasks (for HF/local models)
│   ├── arc.py              # ARC-Easy / ARC-Challenge
│   ├── gsm8k.py            # GSM8K math
│   ├── humaneval.py        # HumanEval code generation
│   ├── mmlu.py             # MMLU
│   └── spellingbee.py
│
├── open_router_tasks/      # OpenRouter chat eval tasks (API-based)
│   ├── common.py           # BaseOpenRouterTask: retry, concurrency, logging
│   ├── gsm8k.py
│   ├── math500.py
│   ├── gpqa_diamond.py
│   ├── arc_challenge.py
│   ├── mbppplus.py
│   └── humaneval_plus.py
│
├── scripts/                # Entry-point scripts
│   ├── base_eval.py        # Base model evaluation (CORE, BPB, sampling)
│   ├── chat_eval.py        # Chat model evaluation (local / HF)
│   ├── chat_eval_open_router.py  # Chat model evaluation via OpenRouter
│   ├── download_hf_model.py
│   ├── run_eval_sweep_hf.sh        # HF model sweep
│   └── run_eval_sweep_open_router.sh  # OpenRouter model sweep
│
├── notebooks/              # Exploratory notebooks and inference scripts
├── tmp/                    # Temporary / scratch files (git-ignored)
├── pyproject.toml
└── uv.lock
```

---

## Base Model Evaluation

`scripts/base_eval.py` evaluates **base (non-chat) models** on three metrics:

| Mode | Description |
|------|-------------|
| `core` | In-context learning accuracy across standard ICL tasks |
| `bpb` | Bits-per-byte on train/val splits |
| `sample` | Free-form generation samples |

### Commands

```bash
# Evaluate a HuggingFace base model across all metrics (multi-GPU)
uv run torchrun --nproc_per_node=8 -m scripts.base_eval \
  --hf-path openai-community/gpt2

# Evaluate a locally trained nano-llama checkpoint
uv run torchrun --nproc_per_node=8 -m scripts.base_eval \
  --model-tag d24 --device-batch-size 16

# Quick single-GPU evaluation (approximate, fewer tokens)
uv run python -m scripts.base_eval \
  --model-tag d24 --device-batch-size 16 \
  --max-per-task 100 --split-tokens 524288

# Evaluate a large model with model parallelism (e.g. 70B)
uv run python -m scripts.base_eval \
  --hf-path meta-llama/Meta-Llama-3-70B \
  --device-map auto --eval core

# Evaluate with 4-bit quantization (requires bitsandbytes)
uv run python -m scripts.base_eval \
  --hf-path meta-llama/Meta-Llama-3-70B \
  --quantize 4bit --eval core
```

Results are written to `$NANOLLAMA_BASE_DIR/base_eval/<model-slug>.csv` (defaults to `out/`).

---

## Chat Model Evaluation (Local / HuggingFace)

`scripts/chat_eval.py` evaluates **instruction-tuned chat models** running locally or loaded via HuggingFace. Tasks are defined in `tasks/`.

### Supported Tasks

`ARC-Easy`, `ARC-Challenge`, `MMLU`, `GSM8K`, `HumanEval`, `SpellingBee`

### Commands

```bash
# Evaluate a locally trained SFT checkpoint on ARC
uv run python -m scripts.chat_eval -i sft -a ARC-Easy

# Multi-GPU evaluation
uv run torchrun --nproc_per_node=8 -m scripts.chat_eval -- -i sft -a ARC-Easy

# Evaluate a HuggingFace chat model
uv run python -m scripts.chat_eval \
  --hf-path meta-llama/Llama-2-7b-chat-hf \
  -a ARC-Easy

# Evaluate on multiple tasks (pipe-separated)
uv run python -m scripts.chat_eval \
  --hf-path meta-llama/Meta-Llama-3-8B-Instruct \
  -a "ARC-Easy|MMLU"
```

---

## Chat Model Evaluation (OpenRouter API)

`scripts/chat_eval_open_router.py` evaluates any model accessible via [OpenRouter](https://openrouter.ai) without requiring local GPU resources. Tasks are defined in `open_router_tasks/`.

### Supported Benchmarks

| Flag | Benchmark | Type |
|------|-----------|------|
| `GSM8K` | Grade School Math 8K | Math reasoning |
| `MATH500` | MATH-500 | Hard math |
| `GPQA_DIAMOND` | GPQA Diamond | PhD-level science |
| `ARC_CHALLENGE` | ARC Challenge | Science MCQ |
| `MBPP_PLUS` | MBPP+ | Python code generation |
| `HUMANEVAL_PLUS` | HumanEval+ | Python code generation |
| `ALL` | All of the above | — |

### Commands

```bash
# Single benchmark, quick smoke test (5 problems)
uv run python -m scripts.chat_eval_open_router \
  -m meta-llama/llama-3.2-3b-instruct \
  -a GSM8K -x 5

# Debug mode: prints one problem's payload, response, and grading
uv run python -m scripts.chat_eval_open_router \
  -m meta-llama/llama-3.2-3b-instruct \
  -a GSM8K --debug

# Multiple benchmarks on one model
uv run python -m scripts.chat_eval_open_router \
  -m meta-llama/llama-3.1-8b-instruct \
  -a GSM8K MATH500 ARC_CHALLENGE \
  --log

# All benchmarks (with both GPQA prompting modes)
uv run python -m scripts.chat_eval_open_router \
  -m meta-llama/llama-3.1-8b-instruct \
  -a ALL \
  --gpqa-mode 0-shot-cot few-shot-cot \
  --log

# Enable reasoning mode (for models that support it)
uv run python -m scripts.chat_eval_open_router \
  -m anthropic/claude-3-7-sonnet \
  -a MATH500 GPQA_DIAMOND \
  --reasoning --log

# Pin to a specific provider (e.g. Together, Fireworks, DeepInfra)
uv run python -m scripts.chat_eval_open_router \
  -m meta-llama/llama-3.1-8b-instruct \
  -a GSM8K \
  --provider Together
```

### Key Flags

| Flag | Default | Description |
|------|---------|-------------|
| `-m` / `--model` | required | OpenRouter model ID |
| `-a` / `--task-name` | `GSM8K` | Benchmark(s) to run |
| `--gpqa-mode` | `0-shot-cot` | GPQA prompting mode(s): `0-shot`, `0-shot-cot`, `few-shot`, `few-shot-cot`, `ALL` |
| `-t` / `--temperature` | `0.0` | Sampling temperature |
| `--max-tokens` | `65536` | Max tokens per response |
| `-x` / `--max-problems` | all | Limit number of problems evaluated |
| `-w` / `--workers` | `10` | Concurrent API workers |
| `--reasoning` | off | Enable reasoning mode |
| `--provider` | auto | Pin to a specific OpenRouter provider |
| `--log` | off | Save raw responses to `.cache/nanollama/<task>/` |
| `--debug` | off | Print one problem end-to-end and exit |
| `--debug-seed` | random | Fix the random seed for `--debug` |

---

## Model Sweep Scripts

Run a full benchmark sweep across a predefined list of models.

### HuggingFace sweep (base model CORE eval)

```bash
HF_TOKEN=hf_... bash scripts/run_eval_sweep_hf.sh
```

Includes automatic OOM retry with decreasing batch size (4 → 2 → 1). Logs per model to `logs/eval_sweep/<model-slug>.log`.

### OpenRouter sweep (all chat benchmarks)

```bash
OPENROUTER_API_KEY=sk-or-... bash scripts/run_eval_sweep_open_router.sh
```

Runs `ALL` benchmarks with `0-shot-cot` and `few-shot-cot` GPQA modes. Logs per model to `logs/eval_sweep_open_router/<model-slug>.log`.

To change which models are swept, edit the `MODELS=(...)` array at the top of the respective script.

---

## Results & Logs

| Output | Location |
|--------|----------|
| Base eval CSVs | `$NANOLLAMA_BASE_DIR/base_eval/<model-slug>.csv` (default: `out/`) |
| OpenRouter raw responses | `.cache/nanollama/<task>/` (when `--log` is set) |
| OpenRouter result summaries | `.cache/nanollama/results/` |
| HF sweep logs | `logs/eval_sweep/<model-slug>.log` |
| OpenRouter sweep logs | `logs/eval_sweep_open_router/<model-slug>.log` |

---

## Downloading HuggingFace Models (offline environments)

```bash
uv run python -m scripts.download_hf_model --hf-path openai-community/gpt2
uv run python -m scripts.download_hf_model --hf-path meta-llama/Llama-3.2-1B
```

This caches all model artifacts under `~/.cache/huggingface/`, which can then be transferred to an air-gapped machine.
