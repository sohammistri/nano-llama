import os
import subprocess
import sys
from pathlib import Path

import pytest


FINEWEB_EDU_HANDLE = "HuggingFaceFW/fineweb-edu"


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
    return env


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
