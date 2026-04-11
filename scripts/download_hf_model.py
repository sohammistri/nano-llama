"""
Download a HuggingFace model and tokenizer to the local cache for offline use.

This script ensures all artifacts needed by base_eval.py are cached locally,
so the entire ~/.cache/huggingface/ directory can be transferred to an
air-gapped environment.

Examples:
    python -m scripts.download_hf_model --hf-path openai-community/gpt2
    python -m scripts.download_hf_model --hf-path meta-llama/Llama-3.2-1B
"""
import os
import argparse

# Set HF_HOME explicitly before importing any HF libraries
os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface/")

from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import scan_cache_dir


def validate_cache(hf_path: str) -> bool:
    """Validate that the downloaded model cache has the expected structure."""
    cache_info = scan_cache_dir()

    # Find the repo matching our model
    matched_repo = None
    for repo in cache_info.repos:
        if repo.repo_id == hf_path and repo.repo_type == "model":
            matched_repo = repo
            break

    if matched_repo is None:
        print(f"ERROR: Model '{hf_path}' not found in cache after download.")
        return False

    # Check for expected subdirectories: blobs/, refs/, snapshots/
    repo_path = Path(matched_repo.repo_path)
    expected_dirs = ["blobs", "refs", "snapshots"]
    missing = [d for d in expected_dirs if not (repo_path / d).is_dir()]

    if missing:
        print(f"ERROR: Missing expected directories in cache: {missing}")
        print(f"  Cache path: {repo_path}")
        return False

    # Report warnings if any
    if cache_info.warnings:
        print(f"WARNING: Cache scan reported {len(cache_info.warnings)} warning(s):")
        for w in cache_info.warnings:
            print(f"  - {w}")

    # Report success with details
    size_mb = matched_repo.size_on_disk / (1024 * 1024)
    size_unit = "MB" if size_mb < 1024 else "GB"
    size_val = size_mb if size_mb < 1024 else size_mb / 1024
    print(f"\nCache validation passed:")
    print(f"  Model:      {hf_path}")
    print(f"  Path:       {repo_path}")
    print(f"  Size:       {size_val:.1f} {size_unit}")
    print(f"  Revisions:  {len(matched_repo.revisions)}")
    print(f"  Structure:  blobs/ refs/ snapshots/ all present")
    return True


def main():
    parser = argparse.ArgumentParser(description="Download a HuggingFace model for offline use")
    parser.add_argument("--hf-path", type=str, required=True, help="HuggingFace model path (e.g. openai-community/gpt2)")
    args = parser.parse_args()

    hf_path = args.hf_path
    print(f"Downloading model: {hf_path}")
    print(f"HF_HOME: {os.environ['HF_HOME']}")

    # Download model weights
    print("\n[1/2] Downloading model weights...")
    AutoModelForCausalLM.from_pretrained(hf_path)
    print("  Model weights downloaded.")

    # Download tokenizer
    print("[2/2] Downloading tokenizer...")
    AutoTokenizer.from_pretrained(hf_path)
    print("  Tokenizer downloaded.")

    # Validate cache
    print("\nValidating cache...")
    ok = validate_cache(hf_path)

    if ok:
        cache_dir = os.environ["HF_HOME"]
        print(f"\nDone! Transfer this directory to your offline environment:")
        print(f"  {cache_dir}")
    else:
        print("\nDownload completed but validation failed. Check the errors above.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
