import os
import subprocess
import sys
from pathlib import Path

import pytest


FINEWEB_EDU_HANDLE = "HuggingFaceFW/fineweb-edu"
SMOKE_MODEL_TAG = "smoke-tiny"


def _smoke_enabled():
    value = os.environ.get("NANOLLAMA_RUN_HF_SMOKE", "")
    return value.lower() in {"1", "true", "yes", "on"}


pytestmark = pytest.mark.skipif(
    not _smoke_enabled(),
    reason="Set NANOLLAMA_RUN_HF_SMOKE=1 to run HuggingFace FineWeb-Edu smoke tests.",
)


def _run_command(command, env, cwd):
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        timeout=600,
    )
    assert result.returncode == 0, (
        f"Command failed: {' '.join(command)}\n"
        f"stdout:\n{result.stdout}\n\n"
        f"stderr:\n{result.stderr}"
    )
    return result


def _smoke_env(base_dir):
    env = os.environ.copy()
    env["NANOLLAMA_BASE_DIR"] = str(base_dir)
    env["NANOLLAMA_DATASET_HANDLE"] = FINEWEB_EDU_HANDLE
    env["NANOLLAMA_TEXT_COLUMN"] = "text"
    env["TORCHDYNAMO_DISABLE"] = "1"
    return env


# [CRITICAL] Exercises HuggingFace data download, tokenizer artifacts, and base training turnover.
def test_fineweb_edu_download_train_and_eval_smoke(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    base_dir = tmp_path / "nanollama-cache"
    env = _smoke_env(base_dir)

    _run_command(
        [
            sys.executable,
            "-m",
            "nanollama.dataset",
            "--dataset-handle",
            FINEWEB_EDU_HANDLE,
            "--text-column",
            "text",
            "--num-files",
            "1",
            "--num-workers",
            "1",
        ],
        env=env,
        cwd=repo_root,
    )

    data_dir = base_dir / "base_data_HuggingFaceFW_fineweb-edu"
    parquet_paths = sorted(data_dir.rglob("*.parquet"))
    assert len(parquet_paths) >= 2
    assert (data_dir / ".nanollama_dataset.json").exists()

    _run_command(
        [
            sys.executable,
            "-m",
            "scripts.tok_train",
            "--max-chars",
            "20000",
            "--doc-cap",
            "1000",
            "--vocab-size",
            "512",
        ],
        env=env,
        cwd=repo_root,
    )

    tokenizer_dir = base_dir / "tokenizer"
    assert (tokenizer_dir / "tokenizer.pkl").exists()
    assert (tokenizer_dir / "token_bytes.pt").exists()

    eval_result = _run_command(
        [sys.executable, "-m", "scripts.tok_eval"],
        env=env,
        cwd=repo_root,
    )

    assert "Comparison with GPT-2" in eval_result.stdout
    assert "Comparison with GPT-4" in eval_result.stdout
    assert (base_dir / "report" / "tokenizer-training.md").exists()
    assert (base_dir / "report" / "tokenizer-evaluation.md").exists()

    train_result = _run_command(
        [
            sys.executable,
            "-m",
            "scripts.base_train",
            "--device-type",
            "cpu",
            "--depth",
            "1",
            "--head-dim",
            "16",
            "--max-seq-len",
            "32",
            "--window-pattern",
            "L",
            "--device-batch-size",
            "1",
            "--total-batch-size",
            "32",
            "--num-iterations",
            "1",
            "--eval-every",
            "-1",
            "--core-metric-every",
            "-1",
            "--sample-every",
            "-1",
            "--save-every",
            "-1",
            "--model-tag",
            SMOKE_MODEL_TAG,
        ],
        env=env,
        cwd=repo_root,
    )

    checkpoint_dir = base_dir / "base_checkpoints" / SMOKE_MODEL_TAG
    assert "step 00000/00001" in train_result.stdout
    assert (checkpoint_dir / "model_000001.pt").exists()
    assert (checkpoint_dir / "meta_000001.json").exists()
    assert (checkpoint_dir / "optim_000001_rank0.pt").exists()
